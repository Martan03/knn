from pathlib import Path
from rich.progress import track
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.loader import IAMStyleDataset, collate_fn_padd
from src.models.style import StyleNet
from src.models.sup_con_loss import SupConLoss


class StyleTrainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.result_dir = Path(args.output)
        self.result_dir.mkdir()
        
        self.model = StyleNet().to(self.device)
        self.loss = SupConLoss().to(self.device)
        
        label_path = args.dataset / "IAM64_train.txt"
        data_path = args.dataset / "IAM64-new/train"
        dataset = IAMStyleDataset(label_path, data_path) 
        self.loader = DataLoader(
            dataset,
            batch_size=args.batch,
            collate_fn=lambda x: collate_fn_padd(x, self.device),
            shuffle=True,
        )
        
        label_path = args.dataset / "IAM64_test.txt"
        data_path = args.dataset / "IAM64-new/test"
        self.test_dataset = IAMStyleDataset(label_path, data_path)
        
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.epochs = args.epochs

    def train(self):
        diff, same = self.test()
        print(f"avg diff: {diff}, avg same: {same}")
        best = diff - same
        for epoch in range(self.epochs):
            loss = self.train_pass()
            print(f"epoch {epoch} loss: {loss}")
            diff, same = self.test()
            if not np.isnan(loss) and diff - same > best:
                best = diff - same
                self.save(self.result_dir / "best.pt")
            print(f"avg diff: {diff}, avg same: {same}")
            self.save(self.result_dir / "last.pt")
    
    def train_pass(self):
        self.model.train()
        loss_sum = 0
        
        for data in track(self.loader, description="training"):
            self.opt.zero_grad()
            x = self.model(data["style"])
            loss = self.loss(x.unsqueeze(1), data["style_label"])
            loss.backward()
            # Clip norm?
            self.opt.step()
            loss_sum += loss.item()
        return loss_sum / len(self.loader)
            

    def save(self, file):
        checkpoint = {
            "model": self.model.state_dict(),
        }
        torch.save(checkpoint, file)
    
    def test(self):
        self.model.eval()
        with torch.no_grad():
            diff_sum = 0
            same_sum = 0
            cnt = 1000
            dlen = len(self.test_dataset)
            for idx in track(self.test_dataset.generator.choice(range(dlen), cnt), description="testing"):
                data = self.test_dataset.get_for_test(idx)
                style = self.model(data["style"].to(self.device))
                same = self.model(data["same"].to(self.device))
                different = self.model(data["different"].to(self.device))
                same_sum += torch.sum(torch.abs(torch.sub(style, same))).item()
                diff_sum += torch.sum(torch.abs(torch.sub(style, different))).item()
            return diff_sum / cnt, same_sum / cnt
        