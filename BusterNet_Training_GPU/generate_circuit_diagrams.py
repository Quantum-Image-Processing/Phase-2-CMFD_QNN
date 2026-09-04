import os
import torch
import pennylane as qml
import matplotlib.pyplot as plt

from vqc_fusion import (
    _amplitude_circuit,
    _reupload_circuit,
    AMP_QUBITS,
    AMP_LAYERS,
    REUP_QUBITS,
    REUP_ROUNDS,
    REUP_LAYERS_PER_ROUND
)

def generate_diagrams():
    output_dir = "circuit_diagram"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving circuit diagrams into: {os.path.abspath(output_dir)}")

    # Dummy inputs & weights
    amp_inputs = torch.randn(9)
    amp_weights = torch.randn(AMP_LAYERS, AMP_QUBITS)

    reup_inputs = torch.randn(9)
    reup_weights = torch.randn(REUP_ROUNDS * REUP_LAYERS_PER_ROUND, REUP_QUBITS)

    styles = ["sketch", "default", "black_white"]

    # 1. Generate Amplitude VQC Diagrams
    for style in styles:
        fig, ax = qml.draw_mpl(_amplitude_circuit, style=style)(amp_inputs, amp_weights)
        fig.suptitle(f"Amplitude VQC Circuit ({AMP_QUBITS} Qubits) - Style: {style}", fontsize=12)
        png_path = os.path.join(output_dir, f"amplitude_vqc_{style}.png")
        svg_path = os.path.join(output_dir, f"amplitude_vqc_{style}.svg")
        fig.savefig(png_path, bbox_inches="tight", dpi=300)
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)

        # Transparent version
        fig, ax = qml.draw_mpl(_amplitude_circuit, style=style)(amp_inputs, amp_weights)
        png_trans = os.path.join(output_dir, f"amplitude_vqc_{style}_transparent.png")
        svg_trans = os.path.join(output_dir, f"amplitude_vqc_{style}_transparent.svg")
        fig.savefig(png_trans, bbox_inches="tight", dpi=300, transparent=True)
        fig.savefig(svg_trans, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print(f"Generated: {png_path}, {svg_path}, {png_trans}, {svg_trans}")

    # 1b. Decomposed (Gate-Level) Amplitude VQC Diagram
    try:
        dec_circuit = qml.transforms.decompose(_amplitude_circuit)
        for style in styles:
            fig, ax = qml.draw_mpl(dec_circuit, style=style)(amp_inputs, amp_weights)
            fig.suptitle(f"Amplitude VQC Decomposed Gate-Level Circuit ({AMP_QUBITS} Qubits) - Style: {style}", fontsize=12)
            png_dec = os.path.join(output_dir, f"amplitude_vqc_decomposed_{style}.png")
            svg_dec = os.path.join(output_dir, f"amplitude_vqc_decomposed_{style}.svg")
            png_dec_trans = os.path.join(output_dir, f"amplitude_vqc_decomposed_{style}_transparent.png")
            svg_dec_trans = os.path.join(output_dir, f"amplitude_vqc_decomposed_{style}_transparent.svg")

            fig.savefig(png_dec, bbox_inches="tight", dpi=300)
            fig.savefig(svg_dec, bbox_inches="tight")
            fig.savefig(png_dec_trans, bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(svg_dec_trans, bbox_inches="tight", transparent=True)
            plt.close(fig)
            print(f"Generated Decomposed: {png_dec_trans} & {svg_dec_trans}")
    except Exception as e:
        print(f"Could not decompose amplitude circuit: {e}")

    # 2. Generate Data Re-uploading VQC Diagrams
    for style in styles:
        fig, ax = qml.draw_mpl(_reupload_circuit, style=style)(reup_inputs, reup_weights)
        fig.suptitle(f"Data Re-uploading VQC Circuit ({REUP_QUBITS} Qubits) - Style: {style}", fontsize=12)
        png_path = os.path.join(output_dir, f"reupload_vqc_{style}.png")
        svg_path = os.path.join(output_dir, f"reupload_vqc_{style}.svg")
        fig.savefig(png_path, bbox_inches="tight", dpi=300)
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)

        # Transparent version
        fig, ax = qml.draw_mpl(_reupload_circuit, style=style)(reup_inputs, reup_weights)
        png_trans = os.path.join(output_dir, f"reupload_vqc_{style}_transparent.png")
        svg_trans = os.path.join(output_dir, f"reupload_vqc_{style}_transparent.svg")
        fig.savefig(png_trans, bbox_inches="tight", dpi=300, transparent=True)
        fig.savefig(svg_trans, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print(f"Generated: {png_path}, {svg_path}, {png_trans}, {svg_trans}")

if __name__ == "__main__":
    generate_diagrams()
