"""Shared figure-saving helper used by the experiment modules."""

import matplotlib.pyplot as plt


def savefig(out, name):
    plt.tight_layout()
    plt.savefig(out / f"{name}.png", dpi=150)
    plt.close()
