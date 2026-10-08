
from __future__ import annotations
import io, json, math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import beam_analysis_ui as ui
from dop_optimizer import (
    OptimizationSettings, analyze, force_envelopes, max_displacement,
    variational_frame_candidate, finite_design_optimize, kkt_verify_design
)
from dop_report import build_report

st.set_page_config(page_title="Đớp Tool — Structural Optimization", page_icon="🏗️", layout="wide")

def _state_df(key, columns):
    df = st.session_state.get(key)
    if df is None:
        return pd.DataFrame(columns=columns)
    return df.copy().reset_index(drop=True)

def _model_from_state():
    nd = _state_df("pf_nd_ed__data", ["x (m)", "y (m)"])
    el = _state_df("pf_el_ed__data", ["i","j","E","A","I","udl_local"])
    sp = _state_df("pf_sup_ed__data", ["node","Loại gối"])
    nl = _state_df("pf_nl_ed__data", ["node","Fx (kN)","Fy (kN)","Mz (kNm)"])

    nodes = [{"x":float(r["x (m)"]), "y":float(r["y (m)"])}
             for _,r in nd.iterrows() if pd.notna(r.get("x (m)")) and pd.notna(r.get("y (m)"))]
    elements = []
    for _,r in el.iterrows():
        if pd.isna(r.get("i")) or pd.isna(r.get("j")): continue
        elements.append({"i":int(r["i"]), "j":int(r["j"]),
                         "E":float(r.get("E",210e6)), "A":float(r.get("A",0.01)),
                         "I":float(r.get("I",1e-4)), "udl_local":float(r.get("udl_local",0) or 0)})
    supports = []
    for _,r in sp.iterrows():
        if pd.isna(r.get("node")): continue
        ux,uy,rz = ui._pf_label_to_bool(r.get("Loại gối","Ngàm"))
        supports.append({"node":int(r["node"]), "ux":ux, "uy":uy, "rz":rz})
    loads=[]
    for _,r in nl.iterrows():
        if pd.isna(r.get("node")): continue
        loads.append({"node":int(r["node"]), "Fx":float(r.get("Fx (kN)",0) or 0),
                       "Fy":float(r.get("Fy (kN)",0) or 0),
                       "Mz":float(r.get("Mz (kNm)",0) or 0)})
    return nodes,elements,supports,loads

def show_model_summary():
    nodes,elements,supports,loads = _model_from_state()
    r = st.session_state.get("pf_result")
    cols = st.columns(5)
    vals=[("Nút",len(nodes)),("Thanh",len(elements)),("Gối",len(supports)),
          ("Tải nút",len(loads)),("FEM","Đã chạy" if r is not None else "Chưa chạy")]
    for c,(a,b) in zip(cols,vals):
        c.metric(a,b)

def render_optimizer():
    st.subheader("🧮 Tối ưu tiết diện bằng biến phân + FEM")
    nodes,elements,supports,loads = _model_from_state()
    if not elements:
        st.warning("Hãy sang tab **Mô hình FEM** để vẽ thanh, gán gối/tải và bấm Solve trước.")
        return

    r = st.session_state.get("pf_result")
    if r is None:
        st.info("Mô hình đã có nhưng chưa có kết quả FEM. Quay lại tab Mô hình FEM và bấm ▶ Solve.")
        return

    c1,c2,c3,c4 = st.columns(4)
    with c1: sigma = st.number_input("σ cho phép (MPa)", 1.0, 1000.0, 235.0, 1.0)
    with c2: dlim = st.number_input("Chuyển vị cho phép (mm)", 0.1, 500.0, 20.0, 0.5)
    with c3: Amin = st.number_input("Amin (mm²)", 1.0, 1e6, 100.0, 10.0)
    with c4: Amax = st.number_input("Amax (mm²)", 100.0, 1e7, 200000.0, 100.0)

    settings=OptimizationSettings(
        sigma_allow=sigma*1000, disp_allow=dlim/1000,
        A_min=Amin/1e6, A_max=Amax/1e6,
        I_min=1e-8, I_max=5e-2
    )
    st.session_state["dop_settings"] = settings
    st.session_state["dop_disp_allow"] = settings.disp_allow

    st.markdown("### 1. Kết quả FEM đầu vào")
    show_model_summary()
    st.write(f"**Chuyển vị lớn nhất hiện tại:** {max_displacement(r)*1000:.4f} mm")

    env=force_envelopes(r)
    df=pd.DataFrame({
        "Thanh":[q["elem_idx"] for q in env],
        "Nmax (kN)":[q["Nmax"] for q in env],
        "Mmax (kNm)":[q["Mmax"] for q in env],
        "A hiện tại (m²)":[elements[q["elem_idx"]]["A"] for q in env],
        "I hiện tại (m⁴)":[elements[q["elem_idx"]]["I"] for q in env],
    })
    st.dataframe(df,use_container_width=True,hide_index=True)

    st.markdown("### 2. Ứng viên biến phân")
    cand,meta=variational_frame_candidate(r,elements,settings)
    cdf=pd.DataFrame({
        "Thanh":range(len(cand)),
        "A*(x) quy đổi (m²)":[x[0] for x in cand],
        "I*(x) quy đổi (m⁴)":[x[1] for x in cand],
        "λA":[x for x in meta["lambdas_A"]]
    })
    st.dataframe(cdf,use_container_width=True,hide_index=True)

    st.latex(r"A^*(x)=\max\left(\frac{|N(x)|}{\sigma_{allow}},\,|N(x)|\sqrt{\lambda/E}\right)")
    st.caption("Ứng viên liên tục được suy ra từ trường nội lực FEM; sau đó được dùng làm điểm khởi tạo cho bài toán tối ưu số có ràng buộc chuyển vị FEM trực tiếp.")

    if st.button("🚀 Tối ưu tiết diện",type="primary",use_container_width=True):
        with st.spinner("Đang tối ưu và phân tích lại FEM..."):
            out=finite_design_optimize(nodes,elements,supports,loads,settings)
        st.session_state["dop_opt"]=out
        st.session_state["dop_kkt"]=None
        st.rerun()

    out=st.session_state.get("dop_opt")
    if out:
        st.markdown("### 3. Phân tích lại sau tối ưu")
        a,b,c=st.columns(3)
        a.metric("V thép trước",f"{out['objective_initial']:.6g} m³")
        b.metric("V thép sau",f"{out['objective_final']:.6g} m³")
        c.metric("Chuyển vị sau",f"{out['d_final']*1000:.4f} mm")
        if out["success"]: st.success(out["message"])
        else: st.warning(out["message"])

        rdf=pd.DataFrame({
            "Thanh":range(len(out["A_final"])),
            "A trước (m²)":out["A_initial"],
            "A tối ưu (m²)":out["A_final"],
            "I trước (m⁴)":out["I_initial"],
            "I tối ưu (m⁴)":out["I_final"],
            "Chiều dài (m)":out["lengths"]
        })
        st.dataframe(rdf,use_container_width=True,hide_index=True)

        rr=out["opt_result"]
        env2=force_envelopes(rr)
        fig=go.Figure()
        for q in env2:
            i=q["elem_idx"]
            fig.add_trace(go.Scatter(x=q["x"],y=q["N"],mode="lines",name=f"E{i} N(x)"))
        fig.update_layout(title="Trường lực dọc sau tối ưu",xaxis_title="x cục bộ (m)",yaxis_title="N (kN)",height=360)
        st.plotly_chart(fig,use_container_width=True)

def render_kkt():
    st.subheader("✅ Kiểm tra KKT")
    out=st.session_state.get("dop_opt")
    if not out:
        st.info("Hãy chạy tối ưu trước.")
        return
    if st.button("🔎 Chạy kiểm tra KKT",type="primary"):
        nodes,elements,supports,loads=_model_from_state()
        settings=st.session_state.get("dop_settings", OptimizationSettings())
        kkt=kkt_verify_design(nodes,elements,supports,loads,settings,out)
        st.session_state["dop_kkt"]=kkt
    kkt=st.session_state.get("dop_kkt")
    if kkt:
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Primal max violation",f"{kkt['primal_max_violation']:.3e}")
        c2.metric("Stationarity",f"{kkt['stationarity_norm']:.3e}")
        c3.metric("Complementarity",f"{kkt['complementarity_residual']:.3e}")
        c4.metric("Dual feasible","YES" if kkt["dual_feasible"] else "NO")
        st.latex(r"\nabla f(x^*)+J_g(x^*)^T\lambda=0,\qquad \lambda\ge0,\qquad g(x^*)\le0,\qquad \lambda_i g_i(x^*)=0")
        st.write("Các đại lượng trên là kiểm tra số tại nghiệm tối ưu. KKT không tự động chứng minh nghiệm toàn cục nếu bài toán không lồi.")

def render_report():
    st.subheader("📄 Thuyết minh")
    out=st.session_state.get("dop_opt")
    kkt=st.session_state.get("dop_kkt")
    if not out:
        st.info("Hãy hoàn thành FEM và tối ưu trước.")
        return
    if kkt is None:
        st.warning("Nên chạy kiểm tra KKT trước khi xuất thuyết minh.")
        return
    nodes,elements,supports,loads=_model_from_state()
    info={"n_nodes":len(nodes),"n_elements":len(elements),"n_supports":len(supports)}
    if st.button("📘 Tạo thuyết minh DOCX",type="primary",use_container_width=True):
        data=build_report(info,out,kkt)
        st.download_button("⬇️ Tải thuyết minh",data=data,file_name="DopTool_ThuyetMinh_ToiUu.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                           use_container_width=True)

def main():
    st.markdown("""
    <style>
    .dop-title{font-size:34px;font-weight:850;letter-spacing:.5px}
    .dop-sub{opacity:.75;font-size:15px}
    </style>
    """,unsafe_allow_html=True)
    st.markdown('<div class="dop-title">🏗️ ĐỚP TOOL</div><div class="dop-sub">Structural Analysis → Variational Optimization → KKT Verification</div>',unsafe_allow_html=True)
    st.divider()

    tabs=st.tabs(["✏️ Mô hình FEM","🧮 Tối ưu biến phân","✅ KKT","📄 Thuyết minh"])
    with tabs[0]:
        ui.render_plane_frame()
    with tabs[1]:
        render_optimizer()
    with tabs[2]:
        render_kkt()
    with tabs[3]:
        render_report()

if __name__=="__main__":
    main()
