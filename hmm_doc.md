## モデル
記号

- $x_t$ : 時刻$t$における観測値
- $z_t$ : 時刻$t$における隠れ変数
- $w_i, |w_i| = c$ : $i$番目の状態
- $\theta$ : モデルパラメータ
- $a_{ij}$ : $w_i$から$w_j$遷移確率
- $b(s_t, x_t)$ : 状態$s_t$における$x_t$の生起確率$p(x_t | s_t)$. 
- $\rho_i$ : 初期状態確率$p(s_1 = w_i)$

## 学習
${\bf x} = [x_1, x_2, \ldots, x_n]^{\top}, {\bf z} = [z_1, z_2, \ldots, z_n]^{\top}$とすると, HMMにおける対数尤度最大化は
$$
\max_{\theta} \log P({\bf x};\theta) = \max_{\theta} \log \sum_{{\bf z}}P({\bf x}, {\bf z};\theta)
$$
のようにlog-sumの形となり直接解くことが難しいため, EMアルゴリズムによって最適化をおこなう. 

イェンゼンの不等式より
$$
\begin{align*}
\log \sum_{{\bf z}}P({\bf x}, {\bf z};\theta) &\geq \sum_{{\bf z}} q({\bf z} |{\bf x} ; \theta_0)\log \frac{P({\bf x}, {\bf z};\theta)}{q({\bf z} |{\bf x} ; \theta_0)}\\
&= Q(\theta, \theta_0) - \sum_{{\bf z}}q({\bf z} |{\bf x} ; \theta_0)\log q({\bf z} |{\bf x} ; \theta_0)
\end{align*}
$$
とできるので$Q(\theta, \theta_0)$を$\theta$について最大化する. 

ここで
$$
\begin{align*}
q({\bf z} |{\bf x} ; \theta_0) &= \frac{P({\bf x}, {\bf z}; \theta_0)}{\sum_{{\bf z}}P({\bf x}, {\bf z}; \theta_0)},\\
Q(\theta, \theta_0)&= \sum_{{\bf z}} q({\bf z} |{\bf x} ; \theta_0)\log P({\bf x}, {\bf z};\theta).
\end{align*}
$$

$$
\begin{align*}
P({\bf x} | {\bf z}; \theta) &= \prod^{n}_{t = 1}b(z_t, x_t),\\
P({\bf z}; \theta) &= P(z_1)a(z_1, z_2)a(z_2, z_3)\cdots a(z_{n - 1}, z_n)
\end{align*}
$$
であることを使って
$$
\begin{align*}
\log P({\bf x}, {\bf z} ; \theta) &= \log P({\bf z}; \theta) + \log P({\bf x} | {\bf z}; \theta)\\
&= \log P(z_1) + \sum^{n - 1}_{t = 1}\log a(z_t, z_{t + 1}) + \sum^n_{t = 1}\log b(z_t, x_t)\\
&= Q(\theta_0, \rho) + Q(\theta_0, A) + Q(\theta_0, \theta_b).
\end{align*}
$$
とできる. 

#### <font color="#0431B4">$Q(\theta_0, A)$の最大化</font>
$$
\begin{align*}
Q(\theta_0, A) &= \sum_{\bf z}P({\bf z} | {\bf x}; \theta_0)\sum^{n - 1}_{t = 1}\log a(z_t, z_{t + 1})\\
&= \sum_{z_1}\cdots \sum_{z_{n}}P(z_1, \ldots, z_n | {\bf x}; \theta_0)\sum^{n - 1}\log a(z_t, z_{t + 1})
\end{align*}
$$
と書ける. $\log a_{ij} ( = \log a(w_i, w_j))$で整理すると
$$
\begin{align*}
Q(\theta_0, A) &= \sum^c_{i = 1}\sum^{c}_{j = 1}\left(\sum^{n - 1}_{t = 1}\sum_{\substack{{\bf z}\\ \substack{(z_t = w_i)}\\ (z_{t + 1} = w_j)}} P(z_1z_2\cdots z_n | {\bf x}; \theta_0)\right)\log a_{ij}
\end{align*}
$$
となり, 
$$
\begin{align*}
\sum_{\substack{{\bf z}\\ \substack{(z_t = w_i)}\\ (z_{t + 1} = w_j)}} P(z_1z_2\cdots z_n | {\bf x}; \theta_0) &= P(z_t = w_i, z_{t + 1} = w_j | {\bf x};\theta_0)\\
&= \xi_t(i, j)
\end{align*}
$$
が成り立つ. ここで
$$\gamma_t(i) = \sum^c_{j} \xi_t(i, j)$$
とすると$Q(\theta_0, A)$を最大にする$a_{ij}$は
$$
\hat{a}_{ij} = \frac{\sum^{n - 1}_{t = 1}\xi_{t}(i, j)}{\sum^{n - 1}_{t = 1}\gamma_t(i)}
$$
と求まる. 

#### <font color="#0431B4">$Q(\theta_0 , \theta_b)$の最大化</font>
$$
\begin{align*}
Q(\theta_0, \theta_b) &= \sum_{{\bf z}} P({\bf z}| {\bf x}; \theta_0)\sum^{n}_{t = 1}\log b(z_t, x_t)\\
&= \sum_{z_1}\cdots\sum_{z_n} P(z_1, \ldots, z_n | x_1, \ldots, x_n; \theta_0)\sum^n_{t = 1}\log b(z_t, x_t)
\end{align*}
$$
である. 隠れ状態が$w_j$の場合だけで整理すれば
$$
\begin{align*}
Q(\theta_0, \theta_b) &= \sum^n_{t = 1} \log b(w_j, x_t) \sum_{\substack{{\bf z}\\(z_t = w_j)}}P(z_1, \ldots, z_n | x_1, \ldots, x_n; \theta_0)\\
&= \sum^n_{t = 1}\gamma_t(j)\log b(w_j, x_t).
\end{align*}
$$
これを状態が$w_j$のときの$x_t$の生成分布のパラメータについて最大化すればよい. 

#### <font color="#0431B4">$Q(\theta_0, \rho)$の最大化</font>
$$
\begin{align*}
Q(\theta_0, \rho) &= \sum_{{\bf z}}P({\bf z} | {\bf x}; \theta_0)\log P(s_1).\\
\end{align*}
$$
$\log P(z_1 = w_i)$について整理すると
$$
\begin{align*}
Q(\theta_0, \rho) &= \sum^c_{i = 1}\left(\sum_{\substack{{\bf z}\\ (z_1 = w_i)}} P(z_1, \ldots, z_n | {\bf x}; \theta_0)\right)\log P(z_1 = w_i)\\
&= \sum^c_{i = 1}\gamma_1(i) \log \rho_i.
\end{align*}
$$
$\sum^c_i \rho_i = 1$と合わせると最適な$\rho_i$は
$$
\begin{align*}
\hat{\rho}_i = \gamma_1(i)
\end{align*}
$$
となる. 

## HMMのスケーリング

長さ$N$の系列, 観測${\bf x}_i,\ i = 1,\ldots, N$, 潜在変数$z_i,\ i = 1,\ldots, N$とする. 

フォワード・バックワードアルゴリズムで求める
$$
\begin{align}
\alpha(z_n) = p({\bf x}_1, \ldots, {\bf x}_n, z_n),\\
\beta(z_n) = p({\bf x}_{n + 1}, \ldots, {\bf x}_N| z_n),
\end{align}
$$
は系列長$N$がそこそこ大きくなるとアンダーフローしてしまう. 

そこでアンダーフローを防ぐスケーリングを行う.

**${\alpha}(z_n)$の場合**

$$
\begin{align}
\hat{\alpha}(z_n) &= p(z_n | {\bf x}_1, \ldots, {\bf x}_n)\\
&= \frac{\alpha(z_n)}{p({\bf x}_1, \ldots, {\bf x}_n)}\\
&= \prod^n_{m = 1}c_m.
\end{align}
$$
ここで
$$
c_n =  p({\bf x}_n | {\bf x}_1, \ldots, {\bf x}_{n - 1}).
$$
$\hat{\alpha}$は$z_n$についての条件付き分布なのでアンダーフローは起こりにくい. 

これを使って$\alpha(z_n)$を求める再帰式の$\hat{\alpha}(z_n)$版が作れる.
$$
\hat{\alpha}(z_n) = {c_n}^{-1}p({\bf x}_n | z_n)\sum_{z_{n - 1}}\hat{\alpha}(z_{n - 1})p(z_n | z_{n -1}).
$$

**${\beta}(z_n)$の場合**

$\alpha$の場合と同様に
$$
\begin{align}
\hat{\beta}(z_n) = \frac{p({\bf x}_{n + 1}, \ldots, {\bf x}_N | z_n)}{p({\bf x}_{n + 1}, \ldots, {\bf x}_N) | {\bf x}_1, \ldots, {\bf x}_n}.
\end{align}
$$

$\hat{\beta}(z_n)$を求める再帰式は
$$
\hat{\beta}(z_n) = {c_{n + 1}}^{-1}\sum_{z_{n + 1}} \hat{\beta}(z_{n + 1})p(x_{n + 1}| z_{n + 1})p(z_{n + 1} | z_n).
$$












