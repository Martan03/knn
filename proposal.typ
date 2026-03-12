#show link: underline

= KNN Project Proposal (Generování obrazu - difuzní modely)

== Vybrané téma

Zadání je docela obecné, existuje spousta způsobů jak využít difuzní modely. My
jsme se rozhodli difuzní modely využít pro generování psaného textu. Vstupem
bude ukázka ručně psaného textu a~text (jiný než na ukázce). Výstupem bude
obrázek tohoto nového textu ve stejném rukopisu jako je ukázka.

== Způsob řešení

Vstup modelu bude obsahovat dvě části - textový obsah, což je text který
se má využít pro generování obrázku obsahující psaný text, a~obrázek obsahující
referenční psaný text, sloužící jako reference stylu písma.

Textový obsah se pošle do kodéru obsahu a referenční psaný text se
pošle do kodéru stylu písma. Jejich výstupy jsou následně smíchány, což je
využito při generování. Následuje samotný odšumovací proces, který začíná
s~gausovským šumem jako vstupem a~postupně jej odšumuje na základě
smíchaných vstupů, které tento proces řídí.

Pro zlepšení extrakce stylu psaní z~co nejméně ukázek textu plánujeme využít
komponentu obsahující vysoké frekvencence z~ukázky textu v podobě
vysokofrekvenčního stylového kodéru.

Implementačním jazykem jsme zvolili Python v kombinaci s knihovnou PyTorch.
Pro extrakci stylu písma využijeme poskytnutý předtrénovaný model. Samotný
difuzní model budeme implementovat pomocí
#link("https://github.com/facebookresearch/DiT")[DiT].

== Hodnocení učení

Pro evaluaci učení z~výsledného obrázku přečteme text pomocí OCR a~vyhodnotíme
Character Error Rate. Dále také extrahujeme styl písma z~výstupního obrázku,
který následně porovnáme s~referenčním textem, což oveří podobnost stylu
písma.

== Experiment

Budeme zkoušet porovnávat kvalitu generování obrázku vůči množství ukázkového
textu. Také vyzkoušíme různé parametry a~hledat, kdy se model bude učit
nejlépe.

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

== GIT repozitář

GIT repozitář pro tento projekt je veřejný na
#link("https://github.com/Martan03/knn")[GitHub].
