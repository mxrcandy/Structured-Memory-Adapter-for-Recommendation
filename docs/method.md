# SMAR method overview

SMAR adapts a Qwen3 recommendation backbone with the backbone parameters frozen. It stores recommendation-specific knowledge in two conditional memory branches:

- **SID Prefix Memory (SPM)** hashes valid local prefixes within the current item's SID. It supplies item-internal code-combination information when predicting later SID codes.
- **Item Transition Memory (ITM)** uses an exact address for eligible prefixes from the most recently completed historical item. It supplies transition information when predicting the next item's first code. The history changes only at item boundaries.

Each branch projects retrieved memory into the backbone hidden size and uses a context gate to control its residual. Ordinary text or invalid SID positions do not activate the corresponding memory. Both branches are trained with the autoregressive recommendation objective while the language-model backbone remains frozen.

This is a concise repository guide, not a replacement for the full paper method specification. See the accompanying submission draft for the complete equations and discussion.
