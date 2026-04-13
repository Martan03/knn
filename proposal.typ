#show link: underline

= KNN Project Proposal (Generování obrazu - difuzní modely)

== Vybrané téma

Zadání je docela obecné, existuje spousta způsobů jak využít difuzní modely. My
jsme se rozhodli difuzní modely využít pro generování psaného textu. Vstupem
bude ukázka ručně psaného textu a~text (jiný než na ukázce). Výstupem bude
obrázek tohoto nového textu ve stejném rukopisu jako je ukázka.

== Způsob řešení

#figure(
    image("model.pdf"),
    caption: [Přehled jak námi využíváný model vypadá.]
)

Vstup modelu bude obsahovat dvě části - textový obsah, což je text který
se má využít pro generování obrázku obsahující psaný text, a~obrázek obsahující
referenční psaný text, sloužící jako reference stylu písma.

Textový obsah se pošle do kodéru obsahu a referenční psaný text se
pošle do kodéru stylu písma. Jejich výstupy jsou následně smíchány, což je
využito při generování. Následuje samotný odšumovací proces, který začíná
s~gausovským šumem jako vstupem a~postupně jej odšumuje na základě
smíchaných vstupů, které tento proces řídí.

Implementačním jazykem jsme zvolili Python v kombinaci s knihovnou PyTorch.
Pro extrakci stylu písma jsme využili předtrénovaný ResNet18. Pro zakódování
písma jsme využili předtrénovaný model RoBERTa. Jako difuzní model jsme
využili upravenou variantu modelu
#link("https://github.com/facebookresearch/DiT")[DiT].

Výstupem modelu RoBERTa je sada vektorů (pro delší vstupy je vektorů více).
Tuto sadu vektorů agregujeme do jednoho vektoru průměrováním komponent vektorů.
Pro delší vstupy by toto mohlo být problémové, protože se zde ztrácí informace
o pozici. Protože ale model trénujeme jen nad jednotlivými slovy, tak by tohle
pro nás neměl být problém.

Obrázek vstupující do Style Encoderu je řádek textu (/slovo) o výšce 64 pixelů
a variabilní šířce. Obrázek je před vstupem do modelu invertován. Toto může být
výhodné, protože původní obrázek je černý text na bílém pozadí. Invertovaný
obrázek je bílý text na černém pozadí. Bílá barva je reprezentována pomocí
hodnoty 1 a černá pomocí hodnoty 0. V invertovaném obrázku je pak tedy aktivace
vysoká v místě, kde se nachází text, narozdíl od původního obrázku, kde je
aktivace vysoká v místech, kde se text nenachází.

Výsledný výstup Style Encoderu a Content Encoderu konkatenujeme do jednoho
vektoru. Tento vektor je zpracován jednou lineární vrstvou a slouží jako
guidance pro difuzní model.

Samotný difuzní model generuje čtvercové obrázky velikosti 256×256 obsahující
daný text na několika řádcích. Každý řádek má výšku 64 pixelů.

#pagebreak()
== Hodnocení učení

Pro evaluaci učení z~výsledného obrázku přečteme text pomocí OCR a~vyhodnotíme
Character Error Rate. Dále také extrahujeme styl písma z~výstupního obrázku,
který následně porovnáme s~referenčním textem, což oveří podobnost stylu
písma. Pro zhodnocení podobnosti výsledných obrázků z obrázky z datasetu
využijeme metodu Fréchet Inception Distance (FID loss).

== Experimenty

Na místě Resnetu bychom rádi zkusili využít jinou síť specializovanou pro
detekci pisatele z ukázky jeho rukopisu.

Aktuálně využíváme předtrénovaný model RoBERTa, který rozdělí text na tokeny,
kde každý token může obsahovat více znaků. Chtěli bychom vyzkoušet využít také
síť, která bude vstupní text rozdělovat znak po znaku.

== Inspirace

Na základě našeho průzkumu již existujících projektů na námi vybrané téma jsme
narazili na projekty #link("https://arxiv.org/pdf/2409.06065")[DiffusionPen],
#link("https://arxiv.org/pdf/2409.04004")[One-DM]
a~#link("https://arxiv.org/pdf/2508.03256")[DiffBrush], které
jsou velmi podobné tomu, co bychom chtěli implementovat. Z~toho důvodu jsou pro
nás tyto projekty inspirací jak samotný model navrhnout.

== Dataset

Našli jsme dva vhodné datasety. Jeden z projektu One-DM dostupný na #link(
    "https://drive.google.com/drive/folders/108TB-z2ytAZSIEzND94dyufybjpqVyn6"
)[google drive]. Tento obsahuje zejména jednotlivá slova. Druhý dataset je z
projektu DiffBrush, dostupný na #link(
    "https://github.com/dailenson/DiffBrush/tree/main/test_data"
)[GitHub]. Tento již obsahuje delší věty.

Aktuálně pro trénování využíváme dataset z One-DM obsahující přes 60000
anglických slov o více než 300 různých rukopisech.

== Aktuální stav

Máme implementovanou iniciální strukturu modelu, kterou jsme se pokusili
natrénovat. Ještě jsme neimplementovali žádný ze způsobů hodnocení učení, ale
je zjevné, že model se nebyl schopen úlohu naučit. Důvodem může být chyba v
implementaci, nevhodné enkodéry nebo nedostatečný čas učení.

== GIT repozitář

GIT repozitář pro tento projekt je veřejný na
#link("https://github.com/Martan03/knn")[GitHub].
