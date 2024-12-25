# Backprop
C = cost
a = activation
z = pre activation
y = desired
w = weight
b = bias

```
y
↓
C ← a₃ ← z₃ ← w₃,₀,₀ a₂,₀
         ↑    w₃,₁,₀ a₂,₁
         b₃   …

a₂,₀ ← z₂,₀ ← w₂,₀,₀ a₁,₀
        ↑     w₂,₁,₀ a₁,₁
       b₂,₀   …

a₂,₁ ← z₂,₁ ← w₂,₀,₁ a₁,₀
        ↑     w₂,₁,₁ a₁,₁
       b₂,₁   …      ^^^^- affects multiple paths
```

## Bias b₂,₀
```
 ∂C     ∂z₂,₀  ∂C
───── = ───── ─────
∂b₂,₀   ∂b₂,₀ ∂z₂,₀

∂z₂,₀
───── = 1
∂b₂,₀
```

## Activation a₁,₀
```
 ∂C     i  ∂z₂,ᵢ ∂a₂,ᵢ  ∂C
───── = Σ  ───── ───── ─────
∂a₁,₀   n₂ ∂a₁,₀ ∂z₂,ᵢ ∂a₂,ᵢ

∂z₂,ᵢ
───── = w₂,₀,ᵢ
∂a₁,₀

∂a₂,ᵢ
───── = ReLU'(∂z₂,ᵢ)
∂z₂,ᵢ
```

## Activation a₂,₀
```
 ∂C      ∂z₃  ∂a₃ ∂C
───── = ───── ─── ───
∂a₂,₀   ∂a₂,₀ ∂z₃ ∂a₃

 ∂z₃
───── = w₃,₀,₀
∂a₂,₀

∂a₃
─── = 1
∂z₃

∂C
─── = 2(a₃ - y)
∂a₃
```
