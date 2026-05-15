#set page(
    paper: "a2",
    margin: 2cm,
)
#show title: set text(size: 50pt, fill: white, weight: "medium")
#show title: set align(center)
#set document(
    title: upper[Generování psaného textu\ pomocí difuzních modelů]
)
#set text(size: 22pt, font: "Lexend")

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
    gutter: 50pt,
    image(width: 100%, "assets/model.pdf"),
    [
        - Vycházíme z DiTu
    ]
)

#grid(
    columns: (1fr, 1fr),
    gutter: 3cm,
    [
        = Experimenty
        Experimenty na textu "hello" a stylu:
        #align(center, image("assets/czechoslovak.png"))
        #grid(
            columns: (1fr, 2fr),
            gutter: 20pt,
            image("assets/1.png"),
            [
                *Verze 1*
                - RoBERTa jako text encoder
                - Kombinování textu i~stylu do vektoru fixní délky
            ]
        )
        #grid(
            columns: (2fr, 1fr),
            gutter: 20pt,
            [
                Druhá iterace modelu.
            ],
            image("assets/2.png")
        )
        #grid(
            columns: (1fr, 2fr),
            gutter: 20pt,
            image("assets/3.png"),
            [
                Třetí iterace modelu.
            ],
        )
        #grid(
            columns: (2fr, 1fr),
            gutter: 20pt,
            [
                Finální iterace modelu.
            ],
            image("assets/final.png")
        )
    ],
    [
        = Dataset
        - Z projektu One-DM
        - Přes 60000 anglických slov od více než 300 pisatelů

        #grid(
            columns: (1fr, 1fr, 1fr),
            gutter: 40pt,
            align: bottom,

            image("assets/c04-110-00-00.png"),
            image("assets/c04-134-02-02.png"),
            image("assets/c04-144-04-00.png"),
        )

        = Dosažené výsledky

        - Text "seems to work"
        - Jednotlivá slova různí pisatelé
        #grid(
            columns: (1fr, 1fr, 1fr),
            align: bottom,

            image("assets/seems.png", width: 100%),
            image("assets/to.png", width: 100%, height: 70pt, fit: "contain"),
            image("assets/work.png", width: 100%),
        )
    ]
)
