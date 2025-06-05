import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- Streamlit setup ---
st.set_page_config(layout="centered")
st.title("💸 Tipping Problem - Fuzzy Logic")
st.divider()
st.header("Fuzzification")

# --- Fuzzy universe definitions ---
quality_range = np.arange(0, 11, 1)
service_range = np.arange(0, 11, 1)
tip_range = np.arange(0, 26, 1)

# --- Membership functions ---
# Quality
quality_low = fuzz.trimf(quality_range, [0, 0, 5])
quality_medium = fuzz.trimf(quality_range, [0, 5, 10])
quality_high = fuzz.trimf(quality_range, [5, 10, 10])

# Service
service_low = fuzz.trimf(service_range, [0, 0, 5])
service_medium = fuzz.trimf(service_range, [0, 5, 10])
service_high = fuzz.trimf(service_range, [5, 10, 10])

# Tip
tip_low = fuzz.trimf(tip_range, [0, 0, 13])
tip_medium = fuzz.trimf(tip_range, [0, 13, 25])
tip_high = fuzz.trimf(tip_range, [13, 25, 25])

# --- User inputs ---
col1, col2 = st.columns(2)
with col1:
    quality_score = st.slider("Rate the *Food Quality*", 0, 10, 5, 1)
with col2:
    service_score = st.slider("Rate the *Service*", 0, 10, 5, 1)

# --- Membership value calculations ---
def get_membership_values(x_range, functions, value):
    return [fuzz.interp_membership(x_range, mf, value) for mf in functions]

quality_membership = get_membership_values(quality_range, [quality_low, quality_medium, quality_high], quality_score)
service_membership = get_membership_values(service_range, [service_low, service_medium, service_high], service_score)

# --- Membership plot function ---
def plot_membership(x, functions, score, title):
    fig, ax = plt.subplots(figsize=(6, 3))

    colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, Green, Red
    labels = ['Bad', 'Decent', 'Great']

    for func, color, label in zip(functions, colors, labels):
        ax.plot(x, func, color=color, linewidth=2.5, label=label, alpha=0.9)

    # Vertical line for the selected score
    ax.axvline(score, color="black", linestyle="--", linewidth=1.5)

    # Minimalist styling
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', labelsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    fig.tight_layout()
    return fig

# --- Show plots and membership values ---
col3, col4 = st.columns(2)
with col3:
    fig = plot_membership(quality_range, [quality_low, quality_medium, quality_high], quality_score, "Food Quality")
    st.pyplot(fig)
    st.subheader("Membership Values (Quality)")
    qc1, qc2, qc3 = st.columns(3)
    qc1.metric("Bad", f"{quality_membership[0]:.2f}")
    qc2.metric("Decent", f"{quality_membership[1]:.2f}")
    qc3.metric("Great", f"{quality_membership[2]:.2f}")

with col4:
    fig = plot_membership(service_range, [service_low, service_medium, service_high], service_score, "Service Quality")
    st.pyplot(fig)
    st.subheader("Membership Values (Service)")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Bad", f"{service_membership[0]:.2f}")
    sc2.metric("Decent", f"{service_membership[1]:.2f}")
    sc3.metric("Great", f"{service_membership[2]:.2f}")

# --- Manual fuzzy inference (rule activation & visualization) ---
# Interpolated values
qual_lo = fuzz.interp_membership(quality_range, quality_low, quality_score)
qual_md = fuzz.interp_membership(quality_range, quality_medium, quality_score)
qual_hi = fuzz.interp_membership(quality_range, quality_high, quality_score)

serv_lo = fuzz.interp_membership(service_range, service_low, service_score)
serv_md = fuzz.interp_membership(service_range, service_medium, service_score)
serv_hi = fuzz.interp_membership(service_range, service_high, service_score)

# Apply rules
active_rule1 = np.fmax(qual_lo, serv_lo)
tip_activation_lo = np.fmin(active_rule1, tip_low)

tip_activation_md = np.fmin(serv_md, tip_medium)

active_rule3 = np.fmax(qual_hi, serv_hi)
tip_activation_hi = np.fmin(active_rule3, tip_high)


# Plot rule activation (manual inference)
def plot_fuzzy_output_activity(x_tip, tip_lo, tip_md, tip_hi,
                                tip_activation_lo, tip_activation_md, tip_activation_hi):
    fig, ax = plt.subplots(figsize=(7, 3))
    tip0 = np.zeros_like(x_tip)

    ax.fill_between(x_tip, tip0, tip_activation_lo, facecolor='#1f77b4', alpha=0.6, label='Rule 1 (Low Tip)')
    ax.plot(x_tip, tip_lo, '--', color='#1f77b4', linewidth=1)

    ax.fill_between(x_tip, tip0, tip_activation_md, facecolor='#2ca02c', alpha=0.6, label='Rule 2 (Med Tip)')
    ax.plot(x_tip, tip_md, '--', color='#2ca02c', linewidth=1)

    ax.fill_between(x_tip, tip0, tip_activation_hi, facecolor='#d62728', alpha=0.6, label='Rule 3 (High Tip)')
    ax.plot(x_tip, tip_hi, '--', color='#d62728', linewidth=1)

    ax.set_title("🔥 Output Membership Activity (Rule Contribution)", fontsize=13, weight='bold')
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(loc='upper right', frameon=False)

    fig.tight_layout()
    return fig


with st.expander("📚 Description of Fuzzy Rules"):
    st.markdown("""
    ### 🧠 Inference Rules Explained

    The fuzzy logic system uses **three simple rules** to determine the appropriate tip based on quality and service:

    **🔵 Rule 1**:  
    *If food quality is bad **OR** service is bad → then tip should be low.*  
    This rule handles poor experiences.

    **🟢 Rule 2**:  
    *If service is decent → then tip should be medium.*  
    This rule ensures acceptable service is rewarded fairly.

    **🔴 Rule 3**:  
    *If food quality is great **OR** service is great → then tip should be high.*  
    This rule captures excellent experiences.

    These rules are combined using fuzzy operators and used to determine the degree of activation of each output level (low, medium, high).
    """)


with st.expander("📊 Rule Activation Output Visualization"):
    fig = plot_fuzzy_output_activity(
        tip_range,
        tip_low, tip_medium, tip_high,
        tip_activation_lo, tip_activation_md, tip_activation_hi
    )
    st.pyplot(fig)


def plot_final_tip_output(x_tip, tip_lo, tip_md, tip_hi, final_tip):
    fig, ax = plt.subplots(figsize=(7, 3))

    # Plot each tip level
    ax.plot(x_tip, tip_lo, color='#1f77b4', linewidth=2.5, label='Low')
    ax.plot(x_tip, tip_md, color='#2ca02c', linewidth=2.5, label='Medium')
    ax.plot(x_tip, tip_hi, color='#d62728', linewidth=2.5, label='High')

    # Vertical line for defuzzified tip
    ax.axvline(final_tip, color='black', linestyle='--', linewidth=2, label=f"Defuzzified Tip: {final_tip:.2f}")

    # Styling
    ax.set_title("Final Tip Output (Defuzzified)", fontsize=13, weight='bold')
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    fig.tight_layout()
    return fig


with st.expander("💡 Tip Inference (Defuzzification)"):
    # Define fuzzy variables
    quality = ctrl.Antecedent(quality_range, 'quality')
    service = ctrl.Antecedent(service_range, 'service')
    tip = ctrl.Consequent(tip_range, 'tip')

    quality['low'] = quality_low
    quality['medium'] = quality_medium
    quality['high'] = quality_high

    service['low'] = service_low
    service['medium'] = service_medium
    service['high'] = service_high

    tip['low'] = tip_low
    tip['medium'] = tip_medium
    tip['high'] = tip_high

    # Define rules
    rule1 = ctrl.Rule(quality['low'] | service['low'], tip['low'])
    rule2 = ctrl.Rule(service['medium'], tip['medium'])
    rule3 = ctrl.Rule(service['high'] | quality['high'], tip['high'])

    tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    tipping_sim = ctrl.ControlSystemSimulation(tipping_ctrl)

    tipping_sim.input['quality'] = quality_score
    tipping_sim.input['service'] = service_score
    tipping_sim.compute()

    st.success(f"💰Recommended Tip: **{tipping_sim.output['tip']:.2f}%**")

    # Tip output plot
    fig = plot_final_tip_output(tip_range, tip_low, tip_medium, tip_high, tipping_sim.output['tip'])
    st.pyplot(fig)
