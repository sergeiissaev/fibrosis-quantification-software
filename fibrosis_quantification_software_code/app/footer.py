# -*- coding: utf-8 -*-

import streamlit as st
from htbuilder import HtmlElement, a, br, div, hr, img, p, styles
from htbuilder.units import percent, px


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):
    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,
    )

    style_hr = styles(display="block", margin=px(8, 8, "auto", "auto"), border_style="inset", border_width=px(2))

    body = p()
    foot = div(style=style_div)(hr(style=style_hr), body)

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Made with ❤️ by ",
        link("https://www.linkedin.com/in/sergei-issaev/", "Sergei Issaev"),
        br(),
        "Github: ",
        link("https://github.com/sergeiissaev/fibrosis-quantification-software", "fibrosis-quantification-software"),
        br(),
        "Contact: sergei740@gmail.com ",
    ]
    layout(*myargs)
