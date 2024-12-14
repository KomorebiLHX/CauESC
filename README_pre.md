# CauESC

This is the repository of our EMNLP 2023 paper CauESC: A Causal Aware Model for Emotional Support Conversation.

The source code will be released after the relevant paper accepted. If your want to make a **human evaluation** with CauESC, the results are available in `cauesc_result`. And **automatic evaluation** can be used as follows.

| Model            |  ACC(%)↑  |   PPL↓    |   R-L↑    |   B-2↑   |   B-3↑   |   B-4↑   |   D-1↑   |   D-2↑    |
| :--------------- | :-------: | :-------: | :-------: | :------: | :------: | :------: | :------: | :-------: |
| Transformer      |     -     |  114.75   |   14.64   |   6.20   |   3.07   |   1.85   |   0.13   |   0.28    |
| MT Transformer   |     -     |  109.44   |   14.94   |   5.99   |   2.60   |   1.37   |   0.15   |   0.35    |
| MoEL             |     -     |   57.03   |   13.93   |   5.48   |   2.40   |   1.25   |   0.74   |   4.12    |
| MIME             |     -     |   56.06   |   14.66   |   6.35   |   2.72   |   1.36   |   0.91   |   5.17    |
| DialoGPT-Joint   |   24.20   |   20.79   |   14.36   |   5.76   |   2.81   |   1.67   |   2.41   |   15.17   |
| Blenderbot-Joint |   31.17   |   18.60   |   17.03   |   5.80   |   3.09   |   1.88   |   3.08   |   13.95   |
| MISC             |   31.63   |   16.16   |   17.91   |   7.31   |   3.78   |   2.20   |   4.41   |   19.71   |
| CauESC           | **33.33** | **15.30** | **18.20** | **8.17** | **4.55** | **2.82** | **4.70** | **19.85** |
