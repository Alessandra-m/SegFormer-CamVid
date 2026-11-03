| Pruning | Shape | Params, M | GMACs | mIoU test before ft | mIoU val (best) | mIoU test | Latency fp32, ms | Latency fp16+compile, ms |
|---|---|---|---|---|---|---|---|---|
| none | original | 3.717 | 4.94 | 0.6912 | 0.7739 | 0.6910 | 8.14 | 3.39 |
| decoder:0.25 | linear_c 192/192/192/192, width 192 | 3.568 | 3.65 | 0.6479 | 0.7691 | 0.6875 | 7.77 | 2.99 |
| decoder:0.5 | linear_c 128/128/128/128, width 128 | 3.453 | 2.71 | 0.2777 | 0.7672 | 0.6906 | 6.93 | 2.61 |
| mlp:0.25 | Mix-FFN hidden 96/192/480/768 | 3.321 | 4.72 | 0.6247 | 0.7626 | 0.6803 | 8.14 | 3.23 |
| heads:0.5,residual4:0.25 | heads 1/1/2/4; stage4 C 192 | 2.877 | 4.62 | 0.5838 | 0.7686 | 0.6850 | 8.17 | 3.33 |
| decoder:0.5,heads:0.5,residual4:0.25 | linear_c 128/128/128/128, width 128; heads 1/1/2/4; stage4 C 192 | 2.621 | 2.40 | 0.2433 | 0.7630 | 0.6775 | 6.91 | 2.53 |
