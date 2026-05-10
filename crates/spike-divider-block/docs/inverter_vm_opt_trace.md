# rlx-eda differentiable CMOS inverter Vm optimization

Circuit: CMOS inverter (NMOS + PMOS via `spike_divider_block::Mosfet`), with input shorted to output so the solved DC operating point IS the switching threshold V_m.

Stimulus: `Vdd = 1.000 V`, `V_m* = 0.600 V`

Loss definition:

$$L = (V_m - V_{m,\text{target}})^2$$

Gradient-driven parameter update (outer Newton on $V_m(Vth_n) - V_m^* = 0$):

$$Vth_n \leftarrow Vth_n - \eta \cdot \frac{V_m - V_m^*}{\partial V_m / \partial Vth_n}$$

## Optimization outcome

- initial: `Vth_n (V) = 0.500`, `V_m = 0.525000`, `loss = 5.625e-3`
- final:   `Vth_n (V) = 0.574`, `V_m = 0.600000`, `loss = 0.000e0`, `steps = 4`

All gradients computed via reverse-mode AD on the rlx graph that stamps the MOSFET LEVEL-1 equations into the MNA residual. No SPICE oracle.

## Rendered charts

| Loss and objective | Parameter evolution |
| --- | --- |
| ![Rendered loss chart](assets/inverter_vm_opt/loss.svg) | ![Rendered parameter chart](assets/inverter_vm_opt/params.svg) |

| Output and error | Gradient signals |
| --- | --- |
| ![Rendered output chart](assets/inverter_vm_opt/output.svg) | ![Rendered gradient chart](assets/inverter_vm_opt/grads.svg) |

## Chart grid

| Row | Left panel | Right panel |
| --- | --- | --- |
| 1 | A. Loss over steps | B. NMOS Vth trajectory |
| 2 | C. V_m tracking vs target | D. ∂V_m / ∂Vth_n evolution |

## A) Loss over steps

```mermaid
xychart-beta
  title "Inverter Vm optimization loss trajectory"
  x-axis "step" [0, 1, 2, 3, 4]
  y-axis "loss"
  line [0.00562501, 0.00097656, 0.00054932, 0.00030899, 0.00000000]
```

Legend:

- line 1: optimization loss $L = (V_m - V_m^*)^2$

## B) NMOS Vth trajectory

```mermaid
xychart-beta
  title "NMOS threshold voltage by step"
  x-axis "step" [0, 1, 2, 3, 4]
  y-axis "Vth_n (V)"
  line [0.5000, 0.5375, 0.5531, 0.5648, 0.5736]
```

Legend:

- line 1: `Vth_n` (NMOS threshold voltage in V)

## C) V_m tracking vs target

```mermaid
xychart-beta
  title "V_m and V_m - V_m_target signed error"
  x-axis "step" [0, 1, 2, 3, 4]
  y-axis "voltage (V)"
  line [0.5250, 0.5688, 0.5766, 0.5824, 0.6000]
  line [-0.0750, -0.0312, -0.0234, -0.0176, 0.0000]
```

Legend:

- line 1: `V_m`
- line 2: `V_m - V_m_target` (signed error)

## D) Gradient evolution

```mermaid
xychart-beta
  title "∂V_m / ∂Vth_n driving the parameter updates"
  x-axis "step" [0, 1, 2, 3, 4]
  y-axis "sensitivity"
  line [1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
```

Legend:

- line 1: $\partial V_m / \partial Vth_n$ (V per unit width-multiplier)

## Step-by-step trace

| step | Vth_n (V) | V_m (V) | loss | dV_m/dVth_n |
| --- | --- | --- | --- | --- |
| 0 | 0.5000 | 0.525000 | 5.6250e-3 | 1.0000e0 |
| 1 | 0.5375 | 0.568750 | 9.7656e-4 | 1.0000e0 |
| 2 | 0.5531 | 0.576563 | 5.4932e-4 | 1.0000e0 |
| 3 | 0.5648 | 0.582422 | 3.0899e-4 | 1.0000e0 |
| 4 | 0.5736 | 0.600000 | 0.0000e0 | 1.0000e0 |
