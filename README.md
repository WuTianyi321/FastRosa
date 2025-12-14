## 项目简介

本项目是一个**研究型实验代码库**，用于研究
[RWKV-LM](https://github.com/BlinkDL/RWKV-LM) 中 **RWKV-8 引入的 ROSA（Rapid Online Suffix Automaton）机制**，重点关注：

> **在保持 ROSA 为确定性、离散、算法模块的前提下，如何对其进行可训练的梯度估计。**

ROSA 是一种基于在线后缀自动机的、无限上下文的符号记忆结构，其前向执行完全是 **离散且不可微的**。本项目尝试在一个最小可控的 toy 任务中，比较不同梯度估计方法在训练此类模块时的行为、效率与效果。

---

## 本项目做了什么

* 实现了一个 **最小可运行的 ROSA 原型**（Python 版，严格离散、确定性）。
* 将 ROSA 通过 **1-bit deterministic gate（`x > 0`）** 嵌入到神经网络中，作为隐藏态的结构性补充信号。
* 在一个合成序列预测任务上，对比两种梯度估计方法：

  * **FLIP（逐 bit 翻转有限差分）**

    * 梯度质量高
    * 计算代价极其昂贵（O(T) 次 ROSA 调用）
  * **RODEO / DisARM 风格的随机梯度估计器**

    * 使用 Bernoulli 采样与相关采样降低方差
    * 每步只需 O(1) 次 ROSA 调用，速度提升约 1–2 个数量级
* 对比并输出：

  * 每步耗时
  * 训练 loss 曲线
  * 最终 next-token accuracy
  * ROSA 本身作为 baseline 的准确率

---

## 与 RWKV-8 / ROSA 的关系

本项目**不是 RWKV-8 的官方实现，也不是完整复现**，而是一个：

* 用于**理解和实验** RWKV-8 中 ROSA 机制的最小研究框架
* 用于探索：

  * ROSA 这类 **确定性、符号化、历史相关模块**
  * 在不依赖注意力、不使用 KV cache 的情况下
  * 如何通过 **非标准梯度估计方法** 与神经网络协同训练

特别地：

* 本项目中的 ROSA **是确定性的**
* 所使用的随机性 **仅存在于梯度估计阶段**，而非前向计算本身

---

## 当前结论（阶段性）

* FLIP 提供了接近“精确”的梯度，但在实际规模下不可用。
* RODEO/DisARM 风格估计器可以在保持 forward 不变的情况下：

  * 将训练速度提升约 **80–100×**
  * 由于FLIP速度过慢，我没有做太多对比实验，但在我的小规模实验上，两者loss相当
* 这验证了：

  * **ROSA 这类离散算法模块使用高效梯度估计器是可训练的**
  * 梯度估计器的设计（偏差 / 方差 / 目标对齐）至关重要

---

## 代码状态说明

* 本代码以**研究清晰性**为优先目标
* 并非高性能实现
* 并未完全实现 RODEO 原论文中的 control variate 训练

---

## 致谢与参考

* RWKV-LM: [https://github.com/BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
* RODEO: *Gradient Estimation with Discrete Stein Operators*

---

## Overview

This repository is a **research-oriented experimental prototype** created to study the **ROSA (Rapid Online Suffix Automaton) mechanism introduced in RWKV-8**, as described in:

👉 [https://github.com/BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)

The focus of this project is **not** to reimplement RWKV-8 itself, but to investigate:

> **How a deterministic, discrete, history-dependent symbolic module like ROSA can be trained inside neural networks using alternative gradient estimators.**

ROSA is fully discrete, non-differentiable, and algorithmic by design. This makes standard backpropagation inapplicable and motivates the exploration of specialized gradient estimation techniques.

---

## What this project does

* Implements a **minimal, fully deterministic ROSA prototype** (Python, suffix automaton–based).
* Integrates ROSA into a neural model via a **1-bit deterministic gate** (`x > 0`).
* Trains the model on a synthetic sequence prediction task.
* Compares two gradient estimators:

  * **FLIP (finite-difference bit flipping)**

    * Near-exact gradients
    * Prohibitively expensive (O(T) ROSA evaluations per step)
  * **RODEO / DisARM-style stochastic estimators**

    * Based on correlated Bernoulli sampling
    * O(1) ROSA evaluations per step, ~1–2 orders of magnitude faster
* Reports:

  * per-step runtime
  * training loss curves
  * final next-token accuracy
  * ROSA baseline accuracy

---

## Relation to RWKV-8 and ROSA

This codebase is **not an official RWKV implementation**.

Instead, it serves as a **controlled research sandbox** to better understand the design space around RWKV-8’s ROSA mechanism:

* ROSA is treated as a **deterministic symbolic memory module**, consistent with RWKV-8’s philosophy.
* All stochasticity appears **only in the gradient estimation process**, not in forward execution.
* The goal is to explore how such modules can be made trainable without attention, KV caches, or differentiable relaxations.

---

## Current findings (preliminary)

* Exact finite-difference gradients (FLIP) are effective but computationally infeasible.
* Stochastic estimators enable **~80–100× speedups** while keeping the forward pass unchanged.
* Due to the slow speed of FLIP, I didn't conduct many experiments. However, in my small-scale experiments, the losses of the two were comparable.

---

## Project status

* Research prototype
* Optimized for clarity and reproducibility, not production use
* Control variate training from the original RODEO paper is not fully implemented

---

## References

* RWKV-LM: [https://github.com/BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
* *Gradient Estimation with Discrete Stein Operators (RODEO)*
