from __future__ import annotations

import streamlit as st


def apply_modern_theme(theme_mode: str = "light") -> None:
    dark = str(theme_mode).strip().lower() == "dark"

    if dark:
        bg_a = "#0e161c"
        bg_b = "#111c24"
        ink = "#e8f1ef"
        muted = "#9db1ac"
        card = "#17242d"
        line = "#29404d"
        shadow = "0 10px 28px rgba(0, 0, 0, 0.35)"
        hero_bg = "linear-gradient(135deg, #18262f 0%, #1a2d38 100%)"
        tab_bg = "#17242d"
        tab_text = "#cfe1dd"
        tab_active_bg = "linear-gradient(135deg, #1f3a47 0%, #3c2e2a 100%)"
        tab_active_text = "#f5fbf9"
        toggle_track_off = "#3b4d58"
        toggle_track_on = "#0ea5a0"
        toggle_knob = "#f8fbfb"
        slider_track = "#324754"
        slider_fill = "#0ea5a0"
        scrollbar_track = "#1b2a34"
        scrollbar_thumb = "#4f6774"
        scrollbar_thumb_hover = "#6e8a98"
        app_bg = (
            "radial-gradient(1200px 450px at 100% -10%, rgba(19, 81, 71, 0.35) 0%, transparent 60%),"
            "radial-gradient(900px 350px at -10% 0%, rgba(154, 52, 18, 0.22) 0%, transparent 55%),"
            "linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 30%)"
        )
    else:
        bg_a = "#f3f7f5"
        bg_b = "#ffffff"
        ink = "#0f2a24"
        muted = "#5f746f"
        card = "#ffffff"
        line = "#d9e6e1"
        shadow = "0 8px 24px rgba(15, 42, 36, 0.08)"
        hero_bg = "linear-gradient(135deg, #ffffff 0%, #f4fbf9 100%)"
        tab_bg = "#ffffff"
        tab_text = "#24433c"
        tab_active_bg = "linear-gradient(135deg, #e0f6f0 0%, #fff3ee 100%)"
        tab_active_text = "#11352f"
        toggle_track_off = "#7f968d"
        toggle_track_on = "#0f766e"
        toggle_knob = "#ffffff"
        slider_track = "#d5e3de"
        slider_fill = "#0f766e"
        scrollbar_track = "#e4ece9"
        scrollbar_thumb = "#88a39a"
        scrollbar_thumb_hover = "#5c7e73"
        app_bg = (
            "radial-gradient(1200px 450px at 100% -10%, #d6efe8 0%, transparent 60%),"
            "radial-gradient(900px 350px at -10% 0%, #fde7df 0%, transparent 55%),"
            "linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 30%)"
        )

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');

        :root {{
            --bg-a: {bg_a};
            --bg-b: {bg_b};
            --ink: {ink};
            --muted: {muted};
            --card: {card};
            --line: {line};
            --shadow: {shadow};
            --radius: 16px;
            --toggle-track-off: {toggle_track_off};
            --toggle-track-on: {toggle_track_on};
            --toggle-knob: {toggle_knob};
            --slider-track: {slider_track};
            --slider-fill: {slider_fill};
            --scrollbar-track: {scrollbar_track};
            --scrollbar-thumb: {scrollbar_thumb};
            --scrollbar-thumb-hover: {scrollbar_thumb_hover};
        }}

        html, body, [class*="css"], p, span, label, h1, h2, h3, h4, h5 {{
            font-family: 'Manrope', sans-serif;
            color: var(--ink);
        }}

        .stApp {{
            background: {app_bg};
        }}

        div.block-container {{
            padding-top: 4.6rem;
        }}

        #theme-switch-anchor + div div[data-testid="stToggle"] {{
            position: fixed;
            top: 0.75rem;
            right: 1rem;
            width: min(320px, calc(100vw - 1.5rem));
            z-index: 9999;
            margin: 0;
        }}

        @media (max-width: 900px) {{
            #theme-switch-anchor + div div[data-testid="stToggle"] {{
                right: 0.5rem;
                top: 0.5rem;
                width: min(260px, calc(100vw - 1rem));
            }}

            div.block-container {{
                padding-top: 4.9rem;
            }}
        }}

        * {{
            scrollbar-width: thin;
            scrollbar-color: var(--scrollbar-thumb) var(--scrollbar-track);
        }}

        *::-webkit-scrollbar {{
            width: 12px;
            height: 12px;
        }}

        *::-webkit-scrollbar-track {{
            background: var(--scrollbar-track);
            border-radius: 999px;
        }}

        *::-webkit-scrollbar-thumb {{
            background: var(--scrollbar-thumb);
            border-radius: 999px;
            border: 2px solid var(--scrollbar-track);
        }}

        *::-webkit-scrollbar-thumb:hover {{
            background: var(--scrollbar-thumb-hover);
        }}

        .app-hero {{
            border: 1px solid var(--line);
            background: {hero_bg};
            border-radius: 20px;
            padding: 1.1rem 1.2rem;
            margin: 0.25rem 0 0.9rem 0;
            box-shadow: var(--shadow);
        }}

        .app-hero h1 {{
            color: var(--ink);
            font-size: 1.65rem;
            line-height: 1.15;
            margin: 0 0 0.3rem 0;
            letter-spacing: -0.02em;
        }}

        .app-hero p {{
            margin: 0;
            color: var(--muted);
        }}

        .mode-pill {{
            display: inline-block;
            margin-top: 0.65rem;
            border-radius: 999px;
            padding: 0.34rem 0.72rem;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            border: 1px solid transparent;
        }}

        .mode-pill.live {{
            background: #d8f3ef;
            color: #0f766e;
            border-color: #9adbcf;
        }}

        .mode-pill.paper {{
            background: #fde7df;
            color: #9a3412;
            border-color: #f4b7a6;
        }}

        div[data-testid="stMetric"] {{
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 0.55rem 0.7rem;
            box-shadow: var(--shadow);
        }}

        div[data-testid="stMetric"] * {{
            color: var(--ink) !important;
        }}

        div[data-testid="stTabs"] button[role="tab"] {{
            border-radius: 12px;
            border: 1px solid var(--line);
            background: {tab_bg};
            margin-right: 0.28rem;
            color: {tab_text};
            font-weight: 700;
        }}

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
            background: {tab_active_bg};
            border-color: var(--line);
            color: {tab_active_text};
        }}

        div[data-testid="stToggle"] {{
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.35rem 0.55rem;
            margin-bottom: 0.35rem;
            box-shadow: var(--shadow);
        }}

        div[data-testid="stToggle"] label,
        div[data-testid="stToggle"] p,
        div[data-testid="stToggle"] span {{
            color: var(--ink) !important;
            font-weight: 700;
        }}

        div[data-testid="stToggle"] [role="switch"] {{
            background-color: var(--toggle-track-off) !important;
            border: 2px solid var(--line) !important;
        }}

        div[data-testid="stToggle"] [role="switch"][aria-checked="true"] {{
            background-color: var(--toggle-track-on) !important;
            border-color: var(--toggle-track-on) !important;
        }}

        div[data-testid="stToggle"] [role="switch"] > div {{
            background: var(--toggle-knob) !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.28);
        }}

        div[data-testid="stToggle"] [data-baseweb="switch"] > div {{
            background-color: var(--toggle-track-off) !important;
            border: 1px solid var(--line) !important;
        }}

        div[data-testid="stToggle"] [data-baseweb="switch"] input:checked + div {{
            background-color: var(--toggle-track-on) !important;
            border-color: var(--toggle-track-on) !important;
        }}

        div[data-testid="stToggle"] [data-baseweb="switch"] > div > div {{
            background: var(--toggle-knob) !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25);
        }}

        div[data-testid="stSlider"] [role="slider"] {{
            background: var(--slider-fill) !important;
            border: 2px solid var(--toggle-knob) !important;
            box-shadow: 0 0 0 2px color-mix(in srgb, var(--slider-fill) 30%, transparent) !important;
        }}

        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {{
            background: var(--slider-track) !important;
        }}

        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) {{
            background: var(--slider-fill) !important;
        }}

        .stButton > button {{
            border-radius: 12px;
            border: 1px solid #a9d3c8;
            background: linear-gradient(135deg, #0f766e 0%, #146a63 100%);
            color: #ffffff;
            font-weight: 700;
        }}

        .stButton > button:hover {{
            border-color: #7fbdae;
            filter: brightness(1.05);
        }}

        .stCodeBlock pre, code, .stTextInput input {{
            font-family: 'IBM Plex Mono', monospace !important;
        }}

        .section-card {{
            border: 1px solid var(--line);
            background: var(--card);
            border-radius: 14px;
            padding: 0.72rem 0.85rem;
            margin: 0.35rem 0 0.65rem 0;
            box-shadow: var(--shadow);
        }}

        .section-card h3 {{
            margin: 0;
            font-size: 1.02rem;
            line-height: 1.25;
            color: var(--ink);
        }}

        .section-card p {{
            margin: 0.24rem 0 0 0;
            color: var(--muted);
            font-size: 0.9rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_card(title: str, subtitle: str = "") -> None:
    st.markdown(
        (
            f'<div class="section-card"><h3>{title}</h3>'
            f"<p>{subtitle}</p></div>"
        ),
        unsafe_allow_html=True,
    )
