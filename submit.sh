#!/usr/bin/bash

NAME=${1:-xsleza26}

typst c proposal.typ
mv proposal.pdf report.pdf
zip -r $NAME.zip src main.py proposal.pdf README.md \
    -x "*/__pycache__" -x "*/__pycache__/*"
