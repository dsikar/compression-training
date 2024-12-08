# Compression Training Results Tables

## Model Performance

| Method | Accuracy (%) | Train Time (s) | Parameters | Compression |
|--------|-------------|----------------|------------|-------------|
| Baseline | 98.2 | 120 | 21,840 | 1.0x |
| DCT | 97.8 | 95 | 16,384 | 3.1x |
| Random Proj | 96.9 | 88 | 16,384 | 3.1x |
| Downsample | 96.5 | 85 | 16,384 | 3.1x |
| Binary Mask | 95.8 | 82 | 16,384 | 3.1x |

## Data Efficiency

| Method | 100% | 75% | 50% | 25% | 10% |
|--------|------|------|------|------|------|
| Baseline | 98.2 | 97.1 | 95.8 | 92.3 | 85.6 |
| DCT | 97.8 | 96.8 | 95.2 | 91.9 | 85.1 |
| Random Proj | 96.9 | 95.7 | 94.1 | 90.8 | 83.9 |

[comment]: # (

\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\hline
Method & Accuracy (\%) & Training Time (s) & Parameters & Compression Ratio \\
\hline
Baseline & 98.2 & 120 & 21,840 & 1.0x \\
DCT & 97.8 & 95 & 16,384 & 3.1x \\
Random Proj. & 96.9 & 88 & 16,384 & 3.1x \\
Downsample & 96.5 & 85 & 16,384 & 3.1x \\
Binary Mask & 95.8 & 82 & 16,384 & 3.1x \\
\hline
\end{tabular}
\caption{Performance metrics across compression methods}
\end{table}

\begin{table}[h]
\centering
\begin{tabular}{lccccc}
\hline
Method & 100\% & 75\% & 50\% & 25\% & 10\% \\
\hline
Baseline & 98.2 & 97.1 & 95.8 & 92.3 & 85.6 \\
DCT & 97.8 & 96.8 & 95.2 & 91.9 & 85.1 \\
Random Proj. & 96.9 & 95.7 & 94.1 & 90.8 & 83.9 \\
\hline
\end{tabular}
\caption{Accuracy (\%) with reduced training data}
\end{table}

)
