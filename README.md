# EITN21 – Project Description, Spring 2026, Part 1
*Fredrik Tufvesson*

---

## Task 1 – Passband Transmitter/Receiver

Implement a passband transmitter/receiver with the following specifications:

| Parameter | Value |
|---|---|
| Base pulse | Half cycle sinus |
| Sampling frequency | 44 100 samples/second |
| Carrier frequency | 4000 Hz |
| Symbol time | 2.2676 ms |
| Modulation format | QPSK |

### Multipath Channel Model

$$r(t) = \sqrt{1 - \alpha^2}\, s(t) + \alpha\, s(t - \mu) + n(t)$$

where:
- $s(t)$ is the transmitted signal
- $n(t)$ is WGN with $\mathbb{E}\{n(t)n(t+\tau)\} = \delta(\tau)N_0/2$
- $\mu = 0.00022676$ seconds
- $\alpha = \dfrac{\text{Group number}}{18}$

Energy should be measured as the total transmitted energy of the passband signal.

**Goal:** Present a plot of bit error probability vs $E_b/N_0$ for error probabilities down to $\approx 10^{-4}$, and compare to theoretical results.

---

## Task 2 – Decode a Mystery Signal

A MATLAB file `Signal.mat` is provided, containing a signal generated according to `SystemModel1.pdf`.

- The message $m$ is of unknown length
- The vector $u$ is the ASCII representation of $m$
- The vector $a$ contains the corresponding QPSK symbols

### QPSK Map

| Bits | Symbol |
|------|--------|
| `00` | $1 + i$ |
| `10` | $-1 + i$ |
| `11` | $-1 - i$ |
| `01` | $1 - i$ |

> If there is an odd number of bits in $u$, a `0` bit is padded at the end.

### Transmitted Symbol Structure

Two pilot symbols $2 + 2i$ are added — one before the data and one after:

```
[ 2+2i | a[1]  a[2]  ...  a[N] | 2+2i ]
  pilot        data symbols       pilot
```

Additional parameters:

| Parameter | Value |
|---|---|
| Pulse shaping | Half cycle sinus, unknown amplitude |
| Symbol time | 2.2676 ms |
| Pulse duration | 2.2676 ms |
| Sampling frequency | 44 100 samples/second |
| Carrier frequency | 4000 Hz |
| Channel | Unknown dispersive + WGN |

**Goal:** Decode the signal and retrieve the secret password.

---

## Task 3 – OFDM with Convolutional Coding

The system model is shown in `SystemModelTask3.pdf`.

### System Parameters

| Parameter | Value |
|---|---|
| Sampling frequency | 44 100 Hz |
| Carrier frequency | 10 000 Hz |
| Number of subcarriers $N_{sc}$ | 128 |
| Cyclic prefix length $N_{cp}$ | 20 |
| OFDM symbol time (excl. CP) | 58 ms |

### Signal Chain

1. Message $m$ (unknown length)
2. ASCII encode → vector $u$
3. Convolutional encode with $(77, 45)$ rate-$\frac{1}{2}$ encoder → vector $v$
   - Encoder starts and ends in the all-zero state
   - Zeros equal to the memory length are appended to $u$ before encoding
4. QPSK map (same map as Task 2) → vector $a$

> If there is an odd number of bits in $v$, a `0` bit is padded at the end.

### Overhead Block (OH)

The transmitted vector $x$ has the following structure:

$$x = \bigl[\underbrace{2p_1 \ 0 \ 2p_2 \ 0 \ \cdots \ 2p_{N_{sc}/2} \ 0}_{\text{Pilots}} \ \Big| \ \underbrace{\tilde{v}}_{\text{length of } m} \ \Big| \ \underbrace{a \ \ 0 \cdots 0}_{\text{multiple of } N_{sc}}\bigr]$$

**Pilots (1st OFDM symbol):**
Every second subcarrier is `0`. The other $N_{sc}/2$ are generated as:

```matlab
randn('state', 100);
P = sign(randn(1, Nsc/2));
x(1:2:end) = 2*P;
```

The zero subcarriers must be estimated by interpolation.

**Length symbol (2nd OFDM symbol):**
- Length $\ell_m$ of $m$ is represented as a 10-bit binary sequence $u_m$
- $u_m$ is encoded by the $(77, 45)$ encoder → $v_m$
- $2N_{sc} - \ell_{v_m}$ zeros are appended so $v_m$ has length $2N_{sc}$
- $v_m$ is QPSK-mapped → $\tilde{v}$ of length $N_{sc}$

**Data symbols:**
Zeros are appended to $a$ so its length is a multiple of $N_{sc}$.

### IFFT and Cyclic Prefix

Partition $x = [x_1 \ x_2 \ x_3 \ \cdots]$ where each block $x_k$ has $N_{sc}$ symbols. Apply IFFT to each block → $y_k$. Concatenate into:

$$y = [\ \underbrace{y_1[1], \ldots, y_1[N_{sc}]}_{y_1} \quad \underbrace{y_2[1], \ldots, y_2[N_{sc}]}_{y_2} \quad \cdots \ ]$$

Then add the cyclic prefix to form $z$:

$$z = [\ \underbrace{y_1[N_{sc}-N_{cp}+1] \cdots y_1[N_{sc}]}_{\text{CP}} \ \underbrace{y_1[1] \cdots y_1[N_{sc}]}_{y_1} \ \underbrace{y_2[N_{sc}-N_{cp}+1] \cdots y_2[N_{sc}]}_{\text{CP}} \ \underbrace{y_2[1] \cdots y_2[N_{sc}]}_{y_2} \ \cdots \ ]$$

### D/A Conversion

Linear interpolation is used: consecutive samples in $z$ are connected by straight lines.

---

## Submission

- **Deadline:** 24 April, 2026
- **Oral presentations:** 4–6 May
- **Format:** PDF, named `PWC26_1_Lastname1_Lastname2.pdf`
- Working alone? Omit `Lastname2`

### Report Requirements

- 3–5 pages
- Block diagram of the receiver
- BER plot from Task 1 with theoretical comparison
- Reliable statistics
- Crisp figures with axis labels (all figures referenced in text)
- Code as plain text in an appendix (no line numbers, no frame)
- Written in proper English

> ⚠️ **Be careful with AI tools.** You must be able to motivate every line of code and may be asked to do simple coding during the oral exam without AI assistance.

**Members of the group are examined individually. All system details must be known by all group members.**