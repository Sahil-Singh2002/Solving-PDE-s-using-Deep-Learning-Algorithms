\section{Solving PDEs with Heaviside Neural Networks}
\label{sec:heaviside-nn}

This repository explores a non--quadrature, closed-form approach to fitting 2D variational PDEs using a shallow neural network with \emph{Heaviside} activations. The key idea is to replace all integrals in the least--squares energy by exact \emph{areas of polygonal intersections} inside the domain, so the mass matrix and right--hand side can be assembled analytically (no sampling).

We study the Poisson model problem on the unit right triangle
\[
-\Delta u(x) = f(x)\ \text{in } \Omega,\qquad 
u=0\ \text{on } \partial\Omega,\qquad
\Omega=\{(x,y): x\ge0,\ y\ge0,\ x+y\le 1\}.
\]

\subsection*{Network parameterisation}
We approximate \(u\) by a piecewise--constant Heaviside network
\[
u_N(x)\;=\;\sum_{i=1}^{N} c_i\,\mathbf 1_{\{w_i^\top x \,\ge\, b_i\}},
\]
with inner parameters \((w_i,b_i)\in\mathbb{R}^2\times\mathbb{R}\) (hyperplanes) and outer coefficients \(c_i\in\mathbb{R}\).
The target \(f\) is also represented as a Heaviside indicator \(f(x)=\mathbf 1_{\{w_f^\top x\ge b_f\}}\) (the constant case \(f\equiv1\) is handled as a limit).

\subsection*{Closed--form assembly}
Let \(H_i=\{x:\,w_i^\top x\ge b_i\}\) and \(H_f=\{x:\,w_f^\top x\ge b_f\}\).
The least--squares energy reduces to areas:
\[
M_{ij} = |\Omega \cap H_i \cap H_j|,\qquad
F_i    = |\Omega \cap H_i \cap H_f|,\qquad
A      = |\Omega \cap H_f| \quad(\text{since } \mathbf 1^2=\mathbf 1).
\]
Minimizing \(J(c)=A-2c^\top F + c^\top M c\) yields the normal equations \(M c=F\).
We solve them robustly by:
(i) custom Cholesky, then (ii) PLU if well--conditioned but not SPD, and (iii) SVD pseudo--inverse as a final fallback.

Optionally, we optimise the inner parameters \((W,B)\) by a reduced objective
\(J^\*(W,B)=A - F^\top M^{-1}F\) using one BFGS step per outer iteration.

\subsection*{Repository layout}
\begin{lstlisting}[basicstyle=\ttfamily\small,frame=single]
Code/
  main.py                 # CLI entry point with multiple demo modes
  experiments.py          # One-call figure generators and study runners
  geometry_utils.py       # Triangular domain (half-spaces, intersections)
  area_utils.py           # Exact area formulas inside Ω
  mass_matrix_utils.py    # Build (M, F, A) for Heaviside target and basis
  solver_utils.py         # Cost, Cholesky/PLU/SVD solvers, PD checks
  hyperplane_utils.py     # Generate hyperplanes + plotting helpers
  optimization.py         # HeavisideObjective: reduced objective + outer loop
  print_utils.py          # Pretty console printing of M and F
  sanity_tests.py         # Analytic area/unit tests for Ω
Pictures/                 # Figures saved by the demos
\end{lstlisting}

\paragraph{Heaviside vs ReLU.}
All derivations and code here use \emph{Heaviside} activations (indicators), \emph{not} ReLU.

\subsection*{Quick start}

\paragraph{Environment.}
\begin{lstlisting}[language=bash,basicstyle=\ttfamily\small,frame=single]
# Python >= 3.9
pip install numpy scipy matplotlib tqdm
\end{lstlisting}

\paragraph{Sanity checks (analytic areas).}
\begin{lstlisting}[language=bash,basicstyle=\ttfamily\small,frame=single]
cd Code
python sanity_tests.py
# Expect: "All sanity tests PASSED ✅"
\end{lstlisting}

\paragraph{Run the demos (\texttt{main.py}).}
\begin{lstlisting}[language=bash,basicstyle=\ttfamily\small,frame=single]
# 1) Small demo: plot hyperplanes, assemble once (target & f ≡ 1)
python main.py --mode demo

# 2) Grid of NN surface plots + ground truth
python main.py --mode surfaces

# 3) Convergence comparison (two targets)
python main.py --mode convergence

# 4) Outer optimisation + diagnostics/contours
python main.py --mode optim --N 2 --wf 1,-1 --bf 0.5 --iters 120 --fit-range 5,100

# Everything in sequence
python main.py --mode all --N 2 --wf 1,-1 --bf 0.5 --iters 120 --fit-range 5,100
\end{lstlisting}

Figures are saved to \texttt{Pictures/} and also displayed.

\subsection*{What each demo does}
\begin{itemize}
  \item \textbf{Demo} (\texttt{--mode demo}): plots the domain with uniform hyperplanes for several \(N\); assembles \((M,F,A)\) for a Heaviside target and for \(f\equiv 1\); solves \(M c=F\); prints diagnostics and positive--definiteness checks.
  \item \textbf{Surfaces} (\texttt{--mode surfaces}): for a set of \(N\), solves once per \(N\), evaluates the NN on a grid over \(\Omega\), and shows a grid of 3D surfaces with the ground truth.
  \item \textbf{Convergence} (\texttt{--mode convergence}): runs two targets (default \(H(y-0.5)\) and \(1\)) and plots log--log error vs neurons with a fitted slope.
  \item \textbf{Optimisation} (\texttt{--mode optim}): runs the outer loop on \((W,B)\); shows loss history, final hyperplanes, order--of--convergence scatter ((\(W,B\)) and \(c\)), and contour slices of the reduced objective:
  \begin{itemize}
    \item \((w_x, w_y)\) plane for up to two neurons,
    \item normalised planes for neuron 0: \((w_x/\|w\|, b/\|w\|)\) and \((w_y/\|w\|, b/\|w\|)\).
  \end{itemize}
\end{itemize}

\subsection*{Reproducing figures via \texttt{experiments.py}}
\begin{lstlisting}[language=Python,basicstyle=\ttfamily\small,frame=single]
from experiments import (
    run_surface_demo,          # grid of surfaces + ground truth
    run_convergence_demo,      # two-target comparison
    run_optimisation_demo      # full optimisation + diagnostics
)

# Example:
run_optimisation_demo(N=2, w_f=[1.0, -1.0], b_f=0.5,
                      num_outer_iter=120, fit_range=(5, 100))
\end{lstlisting}

\subsection*{Notes and tips}
\begin{itemize}
  \item Domains and half--spaces are stored as inequalities \(a^\top x \ge b\); intersections remain convex and areas are computed by clipping.
  \item \texttt{solver\_utils.solve\_for\_c\_custom} tries Cholesky \(\rightarrow\) PLU \(\rightarrow\) SVD automatically with conditioning heuristics.
  \item If you only need Heaviside assembly (no optimisation):
\begin{lstlisting}[language=Python,basicstyle=\ttfamily\small,frame=single]
from mass_matrix_utils import build_mass_matrix_unified
from solver_utils import solve_for_c_custom

M, F, A = build_mass_matrix_unified(domain, W, B, w_f, b_f)
c       = solve_for_c_custom(M, F)
\end{lstlisting}
  \item To change the target \(f\), pass a different \((w_f,b_f)\) to \texttt{build\_mass\_matrix\_unified}; omit them for \(f\equiv 1\).
\end{itemize}

\subsection*{Reference}
A related perspective on neural networks for variational problems:
\begin{quote}
E, Weinan; Yu, Bing. \emph{The Deep Ritz Method: A Deep Learning-Based Numerical Algorithm for Solving Variational Problems}. Communications in Mathematics and Statistics (2018). \url{https://link.springer.com/article/10.1007/s40304-018-0127-z}
\end{quote}

\subsection*{License \& citation}
If you use this code or the generated figures, please cite the repository.  
License: (insert your choice, e.g., MIT/BSD/Apache).
