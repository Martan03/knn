#set page(
    paper: "a2",
    margin: 2cm,
)
#show title: set text(size: 50pt, fill: white, weight: "medium")
#show title: set align(center)
#set document(
    title: upper[Generování psaného textu\ pomocí difuzních modelů]
)
#set text(size: 21pt, font: "Lexend")

#show heading: it => block(
    fill: rgb("#D9D9D9"),
    outset: (left: 1cm, right: 1cm),
    stroke: none,
    width: 100%,
    height: 60pt,
    radius: 10pt,
)[
    #align(center + horizon, text(
        27pt,
        weight: "regular",
        upper(it.body)
    ))
]

#rect(
    fill: rgb("#09A9DF"),
    stroke: none,
    outset: (left: 2cm, right: 2cm, top: 2cm, bottom: 2cm),
)[
    #title()
    #grid(
        align: horizon + center,
        columns: (1fr, 1fr),
        gutter: 50pt,
        align(right)[
            #text(
                27pt,
                fill: white,
                font: "CaskaydiaCove NF",
                [Martin Slezák (xsleza26) \ Jakub Antonín Štigler (xstigl00)]
            )
        ],
        image("assets/fit-logo.png"),
    )
]

#rect(
    fill: rgb("#D9D9D9"),
    stroke: none,
    outset: (left: 1cm, right: 1cm),
    width: 100%,
    height: 60pt,
    radius: 10pt,
    align(center + horizon, text(
        27pt,
        upper[Obecné informace]
    ))
)

#grid(
    columns:(2fr, 1fr),
    gutter: 40pt,
    image(width: 100%, "assets/model.pdf"),
    align(horizon)[
        - Generování zadaného textu se zadaným stylem
        - Modifikovaná verze DiTu
        - Style Encoder realizován konvoluční sítí
        - Content Encoder je tabulka embeddingů jednotlivých znaků
        - Content je připojen přes Cross Attention do DiTu
    ]
)

#grid(
    columns: (1fr, 1fr),
    gutter: 3cm,
    [
        = Experimenty
        Experimenty na textu "hello" a~referenčím stylu:
        #align(center, image("assets/czechoslovak.png"))

        #grid(
            columns: (1fr, 2fr),
            gutter: 20pt,
            align(horizon, image("assets/1.png")),
            [
                *Verze 1*
                - RoBERTa jako text encoder
                - ResNet18 jako style encoder
                - Kombinování textu i~stylu do vektoru fixní délky
                - Málo patchů v~DiT
            ]
        )
        #v(20pt)
        #grid(
            columns: (2fr, 1fr),
            gutter: 20pt,
            [
                *Verze 2*
                - T5 Encoder jako text encoder
                - T5 Encoder se neučí
                - ResNet18 se neučí
                - Více patchů v~DiT
            ],
            align(horizon, image("assets/2.png")),
        )
        #v(20pt)
        #grid(
            columns: (1fr, 2fr),
            gutter: 20pt,
            align(horizon, image("assets/3.png")),
            [
                *Verze 3*
                - Zohlednění masky text embeddingů v~Cross Attention
            ],
        )
        #v(20pt)
        #grid(
            columns: (2fr, 1fr),
            gutter: 20pt,
            [
                *Finální verze*
                - Content Encoder je tabulka embeddingů jednotlivých znaků
                - Style Encoder je námi předtrénovaná konvoluční síť
            ],
            align(horizon, image("assets/final.png")),
        )
    ],
    [
        = Dataset
        - Z projektu One-DM
        - Přes 60000 anglických slov od více než 300 pisatelů

        #v(22pt)
        #grid(
            columns: (1fr, 1fr, 1fr),
            gutter: 40pt,
            align: bottom,

            image("assets/c04-110-00-00.png"),
            image("assets/c04-134-02-02.png"),
            image("assets/c04-144-04-00.png"),
        )
        #v(40pt)

        = Dosažené výsledky

        *Referenční obrázky pro jednotlivá slova:*
        #grid(
            columns: (auto, 1fr, auto),
            align: bottom,
            gutter: 10pt,

            image("assets/seems-style.png", height: 60pt, fit: "contain"),
            image("assets/to-style.png", width: 100%),
            image("assets/work-style.png", height: 60pt),
        )
        #v(25pt)

        *Texty "seems", "to" a "work":*
        #grid(
            columns: (1fr, 1fr, 1fr),
            align: bottom,

            image("assets/seems.png", width: 100%),
            image("assets/to.png", width: 100%, height: 70pt, fit: "contain"),
            image("assets/work.png", width: 100%),
        )
        #v(25pt)

        *Hodnocení modelu:*
        #table(
            columns: (auto, 1fr, 1fr),
            table.header(
                align(center)[*Porovnání stylu*],
                align(center)[*OCR CER*],
                align(center)[*FID*]
            ),
            align(center)[6.5419],
            align(center)[0.9317],
            align(center)[21.8910],
        )
        (pro stejné styly by porovnání stylu mělo být okolo 3.14 a pro rozdílné
        okolo 5.28)
    ]
)
