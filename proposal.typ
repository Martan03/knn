#show link: underline

= KNN Project Proposal (Generování obrazu - difuzní modely)

== Vybrané téma

Zadání je docela obecné, existuje spousta způsobů jak využít difuzní modely. My
jsme se rozhodli difuzní modely využít pro generování psaného textu. Vstupem
bude ukázka ručně psaného textu a~text (jiný než na ukázce). Výstupem bude
obrázek tohoto nového textu ve stejném rukopisu jako je ukázka.

== Existující řešení

Na základě našeho průzkumu již existujících projektů na námi vybrané téma jsme
narazili na projekty #link("https://arxiv.org/pdf/2409.04004")[One-DM],
a~#link("https://arxiv.org/pdf/2508.03256")[DiffBrush], které jsou velmi
podobné tomu, co bychom chtěli implementovat. Z~toho důvodu jsou pro nás tyto
projekty inspirací jak samotný model navrhnout

One-DM funguje nad jednotlivými slovy. Cílem bylo vytvořit model, který dokáže
dobře fungovat i~s~minimální ukázkou stylu na vstupu. Pro zlepšení extrakce
stylu vstupní styl filtrují vysokofrekvenčním filtrem.

Projekt DiffBrush je zaměřený na generování celých vět. Pro správné generování
delších textů ze stylu speciálně extrahuje informaci o~vertikálních
a~horizontálních mezerách v~textu, které poté dodrží při generování.

== Způsob řešení

#figure(
  image("assets/model.pdf"),
  caption: [Přehled jak námi navržený model vypadá.],
)

Vstup modelu obsahuje dvě části - textový obsah, což je text který se má využít
pro generování obrázku obsahující psaný text, a~obrázek obsahující referenční
psaný text, sloužící jako reference stylu písma.

Textový obsah se pošle do kodéru obsahu a referenční psaný text se
pošle do kodéru stylu písma. Zakódovaný styl písma podmiňuje difuzní proces.
Výstup zakódovaného textu je do difuzního procesu předán pomocí Cross
Attention.

Implementačním jazykem jsme zvolili Python v kombinaci s~knihovnou PyTorch.
Pro extrakci stylu písma využíváme konvoluční síť, kterou jsme předtrénovali
pomocí Supervised Contrastive Learning. Pro zakódování písma využíváme
indexovaný embedding jednotlivých znaků. Jako základ pro implementaci samotné
difuze jsme využili #link("https://github.com/facebookresearch/DiT")[DiT].

Obrázek vstupující do Style Encoderu je řádek textu (/slovo) o~výšce 64 pixelů
a~variabilní šířce.

Samotný difuzní model generuje čtvercové obrázky velikosti 256×256 obsahující
daný text na několika řádcích. Každý řádek má výšku 64 pixelů.

== Experimenty

Celkově jsme měli zhruba 15 verzí modelu. Zde jsme vybrali jen relevantní změny
po kterých jsme model trénovali po delší dobu. Pro všechny ukázky je na vstupu
text `hello` a~následující ukázka stylu:

#figure(
    image("assets/czechoslovak.png"),
    caption: [Referenční styl písma využit pro generování]
)

V první iteraci jsme text i~styl kombinovali do vektoru fixní délky, který
podmiňoval difuzní proces. Pro zakódování písma jsme využili nejmenší
předtrénovanou variantu modelu RoBERTa a~pro zakódování stylu písma jsme
využili předtrénovaný model ResNet18. Pro difuzi jsme zvolili konfiguraci s
minimálním počtem patchů. Výsledný model byl velký, pomalý a nefungoval:

#figure(
    image("assets/1.png"),
    caption: [Generovaný text první iterací modelu]
)

V druhé iteraci jsme na místo modelu RoBERTa využili předtrénovaný enkodér
z~modelu T5, u~kterého jsme vypnuli učení. Výsledné textové embeddingy jsme
začali předávat pomocí Cross Attention, kterou jsme do modelu DiT přidali.
Model ResNet18 jsme nechali, ale také jsme u~něj vypnuli učení. Konfiguraci
difuze jsme upravili tak, že jsme zvětšili počet patchů. Výsledný model byl
stále velký a~o~něco pomalejší (kvůli většímu počtu patchů), ale již generoval
obrázky připomínající text ve správném stylu. Samotný text na obrázcích však
byl nečitelný a~jeho délka v~mnoha případech neodpovídala délce požadovaného
textu.

#figure(
    image("assets/2.png"),
    caption: [Generovaný text druhou iterací modelu]
)

V~třetí iteraci jsme upravili Cross Attention tak, aby zohlednil masku
embeddingů. Dosažené výsledky modelu jsou téměř stejné, avšak generovaný text
se zdá mít správnou délku:

#figure(
    image("assets/3.png"),
    caption: [Generovaný text třetí iterací modelu]
)

V~poslední iteraci jsme na místo modelu T5 dali tabulku embeddingů jednotlivých
znaků, kde jednotlivé embeddingy se v~průběhu učí. Na místě ResNet18 jsme
využili vlastní malou konvoluční síť, kterou jsme předtrénovali pomocí
Supervised Contrastive Learning tak, aby pro stejné pisatele dávala na výstup
podobné vektory a~pro různé pisatele vektory rozdílné. Výsledný model je mnohem
menší, podobně rychlý a~generovaný text je již většinou čitelný a~se správným
stylem:

#figure(
    image("assets/final.png"),
    caption: [Generovaný text poslední iterací modelu]
)

== Hodnocení

V~této kapitole jsme pro porovnání ponechali původní výsledky hodnocení první
iterace a~přidali jsme hodnocení druhé iterace.

Pro evaluaci učení jsme využili tři různé metriky: porovnání stylu písma,
Character Error Rate po transkripci textu a FID.

Porovnání stylu písma měří schopnost modelu napodobit rukopis dle referenčního
obrázku. Porovnáváme tedy styl písma z~referenčního obrázku se stylem písma na
vygenerovaném obrázku, kdy metriku počítáme jako sumu absolutních hodnot
rozdílů jednotlivých výstupů neuronové sítě pro extrakci stylu.

OCR CER značí Character Error Rate z~transkripce textu z~generovaného obrázku
s~očekávaným textem. Tato evaluace říká, jak čitelný a~přesný text je modelem
generován. Character Error Rate konkrétně počítá, kolik změn je třeba udělat,
aby texty byly stejné, kdy změna je vložení, smazání nebo substituce znaku.

Pomocí FID, neboli Fréchet Inception Distance, následně spočítáme vzdálenost
generovaných obrázků od obrázků z~celého datasetu, abychom zjistili jejich
podobnost.

=== Hodnocení první iterace

#table(
  columns: (auto, auto, auto),
  table.header([*Porovnání stylu*], [*OCR CER*], [*FID*]),
  [8.615204194188118], [1.170617559523809], [359.0281066894531],
)

Výsledná hodnota porovnání stylu nám vychází okolo `8.615`, což je dokonce více
než průměrná hodnota pro porovnávní stylu písma různých pisatelů, která je
zhruba `5.28`. Očekávaná  hodnota pro porovnání dvou stejných stylů je
v~průměru `3.14`.

Character Error Rate okolo `1.171` je také velmi velká hodnota. Hodnota větší
jak `1` nastane, jelikož OCR predikuje text, který je delší než očekávaný text.
V ideálním případě by se hodnota CER měla pohybovat okolo `0.05` a~méně.

Jako dobré hodnoty pro FID jsou brány hodnoty okolo 50 a~menší, což naše
naměřená hodnota výrazně přesahuje.

Vyhodnocené hodnoty však vzhledem k~aktuálnímu stavu našeho modelu dávají zcela
smysl, jelikož model generuje obrázky velmi podobné šumu.

=== Hodnocení poslední iterace

#table(
  columns: (auto, auto, auto),
  table.header([*Porovnání stylu*], [*OCR CER*], [*FID*]),
  [6.541908099502325], [0.9317171300921294], [21.89108657836914],
)

Výsledná hodnota porovnání stylu nám vyšla `6.542`, což je opět více než
průměrná hodnota pro porovnání stylu písma různých pisatelů. To bude nejspíše
dáno tím, že výsledný obrázek není oříznut do stejného formátu, jako
v~datasetu.

Character Error Rate `0.931` je stále velká hodnota, ale opět to je dáno
formátem obrázku, kdy OCR očekává speficické zarovnání textu, což kvůli
vlastnostem obrázků z~datasetu máme jiné.

FID nám vyšla `21.891`, kdy běžně se hodnoty pod `50` berou jako dobré, což náš
model splňuje.

== Dataset

Pro trénování i~evaluaci jsme využili dataset z~projektu One-DM, který je
dostupný na #link(
    "https://drive.google.com/drive/folders/108TB-z2ytAZSIEzND94dyufybjpqVyn6"
)[google drive]. Tento dataset obsahuje jednotlivá slova. Trénovací část
datasetu obsahuje přes 60000 anglických slov o více než 300 různých rukopisech.

Jednotlivá slova jsou oříznuta tak, aby pokrývala celý obrázek. To způsobuje,
že model generuje slova pokaždé jinak velká, případně linka, na kterou píše, je
jinak vysoko. Toto však pro naše potřeby není problém.

Ukázka obrázků slov z~námi zvoleného datasetu:

#grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 40pt,
    align: bottom,

    figure(
        image("assets/c04-110-00-00.png", width: 100%),
        caption: ["Become"]
    ),
    figure(
        image("assets/c04-134-02-02.png", width: 100%),
        caption: ["stars"]
    ),
    figure(
        image("assets/c04-144-04-00.png", width: 100%),
        caption: ["success"]
    )
)

== Natrénovaný model

Finální model jsme trénovali na celém datasetu na 32 epoch, kde jedna
epocha trvala zhruba 25 minut (okolo 13 hodin a 20 minut). Loss hodnoty měli
během učení tendency klesat, což lze pozorovat v~následujícím grafu.

#figure(image("assets/loss.png"))

Ukázka vygenerovaných slov od různých pisatelů:

#grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 40pt,
    align: bottom,

    figure(
        image("assets/seems.png", width: 100%),
        caption: ["seems"]
    ),
    figure(
        image("assets/to.png", width: 100%, height: 64pt, fit: "contain"),
        caption: ["to"]
    ),
    figure(
        image("assets/work.png", width: 100%),
        caption: ["work"]
    )
)

== Shrnutí

Při návrhu modelu velmi záleží na jeho struktuře. Často je mnohem lepší využít
jednoduché specializované části než obecné předtrénované modely. Náš první
model obsahoval mnoho samostatně kvalitních částí, které však byly špatně
poskládané dohromady, a~ve výsledku se model nenaučil, co jsme chtěli.

Náš poslední model je již mnohem menší a~jednodušší. Jednotlivé části byly
specializované na svoji činnost a~model má na první pohled výrazně lepší
výsledky.

Výsledný model je schopný generovat většinou čitelná samostatná slova v~zadaném
stylu. Není však schopný generovat delší úseky textu. Důvodem je trénovací
dataset, který obsahuje pouze jednotlivá slova nebo interpunkci. Model tak
nikdy nebyl učen na znacích jako je například mezera a~tudíž neví, co s~nimi
dělat. V~případě jiného datasetu je pravděpodobné, že model by se naučil
generovat i~delší úseky textu.

== GIT repozitář

GIT repozitář pro tento projekt je veřejný na
#link("https://github.com/Martan03/knn")[GitHub].
