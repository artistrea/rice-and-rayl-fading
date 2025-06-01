import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib

font = {'size'   : 14}
matplotlib.rc('font', **font)

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

def get_cdf(samples):
    values, counts = np.unique(samples, return_counts=True)
    cum_counts = np.cumsum(counts)
    normalized = cum_counts / cum_counts[-1]
    return values, normalized


def marcumq(a, b):
    return scipy.stats.ncx2.sf(b**2, 2, a**2)


def gen_rayleigh_va(rng, Omega_c, N):
    # 2 sigma^2 = Omega_c
    sigma = np.sqrt(Omega_c/2)
    x = rng.normal(0, sigma, N)
    y = rng.normal(0, sigma, N)

    return np.sqrt(x**2 + y**2)

def verify_rayleigh_mom(va, Omega_c):
    va_mom = []
    for n in range(1, 11):
        va_mom.append(
            np.mean(va ** n)
        )
    th_x = np.linspace(1, 10, 100)
    th_mom = (Omega_c)**(th_x/2)  * scipy.special.gamma(1 + th_x/2)

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.semilogy(th_x, th_mom, color="red", label="Momentos Téoricos")
    th_x_specific = np.arange(1, 11)
    th_mom_specific = (Omega_c)**(th_x_specific/2)  * scipy.special.gamma(1 + th_x_specific/2)
    ax.scatter(
        th_x_specific, th_mom_specific,
        marker='o',
        s=50,
        facecolors='none',
        edgecolors='red',
        linewidths=1.5,
    )

    ax.scatter(
        th_x_specific, va_mom,
        marker='x',
        s=75,
        facecolors='black',
        label="Momentos Empíricos"
    )
    ax.set_xlabel(r"Ordem dos momentos - $n$")
    ax.set_ylabel(r"Momentos estatísticos")
    ax.set_title(rf"$\Omega_c = {Omega_c}$, $N  = 10^{int(np.log10(len(va)))}$")
    # ax.set_ylim((0.1, 150))
    ax.set_xlim((1, 10))
    ax.grid(True, alpha=0.2, color="gray")

    ax.legend()
    return fig


def verify_rayleigh_pdf(va, Omega_c, M):
    x = np.linspace(0, 3.5, 1000)
    p = 2 * x * np.exp(-x * x / Omega_c) / Omega_c

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.hist(
        va, density=True, bins=M, edgecolor="black", label="Histograma"
    )
    ax.plot(x, p, color="red", label="PDF Teórica")
    ax.set_ylim((0.0, 1.0))
    ax.set_xlim((0, 3.5))
    ax.set_xlabel(r"Envoltória do Canal - $\beta$")
    ax.set_ylabel(r"FDP - $p_\beta(\beta)$")
    ax.set_title(rf"$\Omega_c = {Omega_c}$, $N  = 10^{int(np.log10(len(va)))}$, $M = {M}$")
    ax.grid(True, alpha=0.2, color="gray")

    ax.legend()
    return fig


def verify_rayleigh_cdf(va, Omega_c, M):
    x = np.linspace(0, 3.5, 200)
    cdf = 1 - np.exp(-x * x / Omega_c)

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(
        x, cdf,
        linestyle="-.",
        color="red",
        label="CDF Teórica"
    )
    ax.scatter(
        x[::10], cdf[::10],
        marker='s',
        s=50,
        facecolors='none',
        edgecolors='red',
        linewidths=1.5,
    )

    va_x, va_cdf = get_cdf(va)
    ax.plot(
        va_x, va_cdf,
        linestyle="--",
        color="blue",
        label="CDF Empírica"
    )

    ax.grid(True, alpha=0.2, color="gray")

    ax.set_xlabel(r"Envoltória do Canal - $\beta$")
    ax.set_ylabel(r"FDA - $p_\beta(\beta)$")
    ax.set_title(rf"$\Omega_c = {Omega_c}$, $N  = 10^{int(np.log10(len(va)))}$, $M = {M}$")
    ax.set_ylim((0.0, 1.0))
    ax.set_xlim((0, 3.5))

    ax.legend()
    return fig


def gen_rice_va(rng, Omega_c, K_r, N):
    # mu**2 = mu_x**2 + mu_y**2
    # K_r = mu**2 / (2 * sigma**2)
    #  => mu**2 = K_r * (2 * sigma**2)
    # Omega_c = mu**2 + 2 * sigma**2
    #  => Omega_c = K_r * (2 * sigma**2) + 2 * sigma**2
    #     Omega_c = (K_r + 1) * (2 * sigma**2)
    #     Omega_c / (2 * (K_r + 1)) =  sigma**2
    sigma = np.sqrt(
        Omega_c / (2 * (K_r + 1))
    )
    mu = np.sqrt(
        K_r * 2
    ) * sigma
    mu_y = mu_x = mu / np.sqrt(2)

    x = rng.normal(mu_x, sigma, N)
    y = rng.normal(mu_y, sigma, N)

    return np.sqrt(x*x + y*y)

def verify_rice_pdf(va, Omega_c, K_r, M):
    x_lim = 3.5
    x = np.linspace(0, x_lim, 1000)
    p = 2 * x * (K_r + 1) *\
        np.exp(-K_r - (K_r+1)*x*x / Omega_c)\
        * np.i0(2 * x * np.sqrt(K_r * (K_r + 1) / Omega_c))\
        / Omega_c

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.hist(
        va, density=True, bins=M, edgecolor="black", label="Histograma",
    )
    ax.plot(x, p, color="red", label="PDF Teórica")
    mxy = np.max(p)
    ax.set_ylim((0.0, np.max([1.0, mxy + 0.1])))
    ax.set_xlim((0, x_lim))
    ax.set_xlabel(r"Envoltória do Canal - $\beta$")
    ax.set_ylabel(r"FDP - $p_\beta(\beta)$")
    ax.set_title(rf"$\Omega_c = {Omega_c}$, $K_R = {K_r}$, $N  = 10^{int(np.log10(len(va)))}$, $M = {M}$")
    ax.grid(True, alpha=0.2, color="gray")

    ax.legend()
    return fig


def verify_rice_cdf(va, Omega_c, K_r, M):
    x = np.linspace(0, 3.5, 200)
    cdf = 1 - marcumq(np.sqrt(2 * K_r), x * np.sqrt(2 * (K_r + 1) / Omega_c))

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(
        x, cdf,
        linestyle="-.",
        color="red",
        label="CDF Teórica"
    )
    ax.scatter(
        x[::10], cdf[::10],
        marker='s',
        s=50,
        facecolors='none',
        edgecolors='red',
        linewidths=1.5,
    )

    va_x, va_cdf = get_cdf(va)
    ax.plot(
        va_x, va_cdf,
        linestyle="--",
        color="blue",
        label="CDF Empírica"
    )

    ax.grid(True, alpha=0.2, color="gray")

    ax.set_ylim((0.0, 1.0))
    ax.set_xlim((0, 3.5))
    ax.set_xlabel(r"Envoltória do Canal - $\beta$")
    ax.set_ylabel(r"FDA - $p_\beta(\beta)$")
    ax.set_title(rf"$\Omega_c = {Omega_c}$, $K_R = {K_r}$, $N  = 10^{int(np.log10(len(va)))}$, $M = {M}$")

    ax.legend()
    return fig


def verify_rice_mom(va, Omega_c, K_r):
    va_mom = []
    for n in range(1, 11):
        va_mom.append(
            np.mean(va ** n)
        )
    th_x = np.linspace(1, 10, 100)
    th_mom = []
    for t_x in th_x:
        mom = (Omega_c / (K_r + 1))**(t_x/2)  * scipy.special.gamma(1 + t_x/2)\
            * scipy.special.eval_genlaguerre(t_x/2, 0, -K_r)
        th_mom.append(mom)

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.semilogy(th_x, th_mom, color="red", label="Momentos Téoricos")
    th_x_specific = np.arange(1, 11)
    th_mom_specific = []
    for t_x in th_x_specific:
        mom = (Omega_c / (K_r + 1))**(t_x/2)  * scipy.special.gamma(1 + t_x/2)\
            * scipy.special.eval_genlaguerre(t_x/2, 0, -K_r)
        th_mom_specific.append(mom)
    ax.scatter(
        th_x_specific, th_mom_specific,
        marker='o',
        s=50,
        facecolors='none',
        edgecolors='red',
        linewidths=1.5,
    )

    ax.scatter(
        th_x_specific, va_mom,
        marker='x',
        s=75,
        facecolors='black',
        label="Momentos Empíricos"
    )
    # ax.set_ylim((0.1, 150))
    ax.set_xlim((1, 10))
    ax.set_xlabel(r"Ordem dos momentos - $n$")
    ax.set_ylabel(r"Momentos estatísticos")
    ax.set_title(rf"$\Omega_c = {Omega_c}$, $K_R = {K_r}$, $N  = 10^{int(np.log10(len(va)))}$")
    ax.grid(True, alpha=0.2, color="gray")

    ax.legend()
    return fig

rng = np.random.default_rng(103)

for Omega_c in [1,2,3]:
    va = gen_rayleigh_va(rng, Omega_c, int(1e7))

    f = verify_rayleigh_pdf(va, Omega_c, 80)
    f.tight_layout()
    f.savefig(f"figs/ray-pdf-{Omega_c}.png")
    plt.close()

    f = verify_rayleigh_cdf(va, Omega_c, 80)
    f.tight_layout()
    f.savefig(f"figs/ray-cdf-{Omega_c}.png")
    plt.close()

    f = verify_rayleigh_mom(va, Omega_c)
    f.tight_layout()
    f.savefig(f"figs/ray-mom-{Omega_c}.png")
    plt.close()

# K_r = 10
i=0
for Omega_c, K_r in zip([1,1,3,3], [1, 10, 1, 10]):
    i+=1
    va = gen_rice_va(rng, Omega_c, K_r, int(1e7))

    f = verify_rice_pdf(va, Omega_c, K_r, 80)
    f.tight_layout()
    f.savefig(f"figs/rice-pdf-{i}.png")
    plt.close()

    f = verify_rice_cdf(va, Omega_c, K_r, 80)
    f.tight_layout()
    f.savefig(f"figs/rice-cdf-{i}.png")
    plt.close()
    
    f = verify_rice_mom(va, Omega_c, K_r)
    f.tight_layout()
    f.savefig(f"figs/rice-mom-{i}.png")
    plt.close()
    

# plt.tight_layout()
# plt.show()

# Os gráficos de PDF, CDF e Momentos podem ser vistos
# nas Figuras \ref{fig:rice-pdf-1}, \ref{fig:rice-cdf-1}, \ref{fig:rice-mom-1} para o caso com $\Omega = 1$ e $K_R = 1$,
# nas Figuras \ref{fig:rice-pdf-2}, \ref{fig:rice-cdf-2}, \ref{fig:rice-mom-2} para o caso com $\Omega = 1$ e $K_R = 10$,
# nas Figuras \ref{fig:rice-pdf-3}, \ref{fig:rice-cdf-3}, \ref{fig:rice-mom-3} para o caso com $\Omega = 3$ e $K_R = 1$, e
# nas Figuras \ref{fig:rice-pdf-4}, \ref{fig:rice-cdf-4}, \ref{fig:rice-mom-4} para o caso com $\Omega = 3$ e $K_R = 10$.

# \begin{figure}
#     \centering
#     \includegraphics[width=0.5\linewidth]{figs/rice-pdf-1.png}
#     \caption{PDF da envoltória Rice, Caso 1}
#     \label{fig:rice-pdf-1}
# \end{figure}
# \begin{figure}
#     \centering
#     \includegraphics[width=0.5\linewidth]{figs/rice-pdf-2.png}
#     \caption{PDF da envoltória Rice, Caso 2}
#     \label{fig:rice-pdf-2}
# \end{figure}

# \begin{figure}
#     \centering
#     \includegraphics[width=0.5\linewidth]{figs/rice-pdf-3.png}
#     \caption{PDF da envoltória Rice, Caso 3}
#     \label{fig:rice-pdf-3}
# \end{figure}

# \begin{figure}
#     \centering
#     \includegraphics[width=0.5\linewidth]{figs/rice-pdf-4.png}
#     \caption{PDF da envoltória Rice, Caso 4}
#     \label{fig:rice-pdf-4}
# \end{figure}

# \begin{figure}
#     \centering
#     \includegraphics[width=0.5\linewidth]{figs/rice-cdf-1.png}
#     \caption{CDF da envoltória Rice, Caso 1}
#     \label{fig:rice-cdf-1}
# \end{figure}

# \begin{figure}
#     \centering
#     \includegraphics[width=0.5\linewidth]{figs/rice-cdf-2.png}
#     \caption{CDF da envoltória Rice, Caso 2}
#     \label{fig:rice-cdf-2}
# \end{figure}


# \begin{figure}
#     \centering
#     \includegraphics[width=0.5\linewidth]{figs/rice-cdf-3.png}
#     \caption{CDF da envoltória Rice, Caso 3}
#     \label{fig:rice-cdf-3}
# \end{figure}

# \begin{figure}
#     \centering
#     \includegraphics[width=0.5\linewidth]{figs/rice-cdf-4.png}
#     \caption{CDF da envoltória Rice, Caso 4}
#     \label{fig:rice-cdf-4}
# \end{figure}


# \begin{figure}
#     \centering
#     \includegraphics[width=0.5\linewidth]{figs/rice-mom-1.png}
#     \caption{Momentos da envoltória Rice, Caso 1}
#     \label{fig:rice-mom-1}
# \end{figure}
# \begin{figure}
#     \centering
#     \includegraphics[width=0.5\linewidth]{figs/rice-mom-2.png}
#     \caption{Momentos da envoltória Rice, Caso 2}
#     \label{fig:rice-mom-2}
# \end{figure}
# \begin{figure}
#     \centering
#     \includegraphics[width=0.5\linewidth]{figs/rice-mom-3.png}
#     \caption{Momentos da envoltória Rice, Caso 3}
#     \label{fig:rice-mom-3}
# \end{figure}

# \begin{figure}
#     \centering
#     \includegraphics[width=0.5\linewidth]{figs/rice-mom-4.png}
#     \caption{Momentos da envoltória Rice, Caso 4}
#     \label{fig:rice-mom-4}
# \end{figure}

