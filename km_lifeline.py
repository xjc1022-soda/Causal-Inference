from lifelines import KaplanMeierFitter
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import torch


def plot_kmf(high_data,low_data,split):
    os_high, event_high = high_data[0], high_data[1]
    df_high = pd.DataFrame(list(zip(os_high, event_high)), columns=['T','E'])
    kmf_1 = KaplanMeierFitter(label="high scores")
    kmf_1.fit(df_high['T'],event_observed=df_high['E'])

    os_low, event_low = low_data[0], low_data[1]
    df_low = pd.DataFrame(list(zip(os_low, event_low)), columns=['T','E'])
    kmf_2 = KaplanMeierFitter(label="low scores")
    kmf_2.fit(df_low['T'],event_observed=df_low['E'])

    kmf_1.plot(at_risk_counts=True, show_censors=False, color='coral')
    kmf_2.plot(at_risk_counts=True, show_censors=False, color='#054E9F')
    plt.savefig("Km_line_"+split)
    plt.clf()

    kmf_1.plot(at_risk_counts=True, show_censors=False, color='coral')
    plt.savefig("Km_line_"+split+"_high")
    plt.clf()
    kmf_2.plot(at_risk_counts=True, show_censors=False, color='#054E9F')
    plt.savefig("Km_line_"+split+"_low")
    plt.clf()

