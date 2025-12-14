# WaterMarking
WaterMarking-project
# Digital Image Watermarking System  
### An AI-Assisted, Modular, and Reproducible Implementation

---

## 1. Motivation

In the digital era, images can be copied, modified, and redistributed at extremely low cost.  
As a result, **copyright infringement, unauthorized reuse, and loss of content ownership** have become increasingly serious issues.

This project aims to implement a **basic yet complete digital image watermarking system** that embeds creator information invisibly into an image, enabling **copyright protection, image authentication, and content traceability**.

Beyond functionality, this project emphasizes **modern software engineering practices**, including:
- Modular system design  
- AI-assisted development  
- Cross-device collaboration  
- Reproducible experiments  

---

## 2. Project Objectives

The objectives of this project are:

1. To design a **binary watermark** (e.g., logo or ID) suitable for low-bit embedding  
2. To embed the watermark invisibly into a cover image using **bit-plane replacement**  
3. To enable **reversible extraction** using XOR-based logic (non-blind watermarking)  
4. To evaluate:
   - **Imperceptibility** using PSNR  
   - **Robustness** under common image attacks (JPEG compression, Gaussian noise)  
5. To demonstrate a **modular, AI-friendly development workflow** suitable for team collaboration  

---

## 3. Core Techniques

### 3.1 Bit-Plane Embedding (Spatial Domain)

- The watermark is embedded into a **low bit plane** of image pixels
- Modifying low-order bits minimizes visual distortion
- The system supports selecting:
  - Bit plane index (e.g., LSB, 2nd LSB)
  - Color channel (B, G, or R)

---

### 3.2 XOR-Based Embedding and Extraction

To ensure reversibility and clean extraction logic, the system adopts **XOR-based embedding**:

\[
b' = b \oplus w
\]

Where:
- \( b \) is the original bit in the selected bit plane  
- \( w \) is the watermark bit  
- \( b' \) is the embedded bit  

Extraction is performed by:

\[
w = b \oplus b'
\]

This design guarantees correct recovery in non-blind extraction and simplifies system integration.

---

### 3.3 Evaluation Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**  
  Measures visual imperceptibility between the original image and the watermarked image.

- **BER (Bit Error Rate)**  
  Measures the accuracy of recovered watermark bits under different attacks.

---

## 4. System Architecture

The system is designed using an **interface-oriented, modular architecture**:
Watermark Design
↓
Bitstream Encoding
↓
XOR Bit-Plane Embedding
↓
Attack Simulation
↓
Watermark Extraction
↓
Evaluation (PSNR / BER)


Each module:
- Has a clearly defined responsibility
- Can be developed independently on different devices
- Communicates only via defined input/output contracts

---

## 5. Project Structure
```
digital-watermarking-project/
├── config/
│ └── config.yaml # Global system configuration
│
├── src/
│ ├── watermark_design/ # Watermark generation & encoding
│ ├── embedding/ # XOR-based bit-plane embedding
│ ├── extraction/ # Watermark extraction & decoding
│ ├── attacks/ # JPEG & noise attack simulation
│ ├── evaluation/ # PSNR & BER metrics
│ └── main.py # System integration pipeline
│
├── data/
│ ├── cover/ # Cover images
│ └── watermark/ # Watermark logo
│
├── output/
│ ├── stego/ # Watermarked images
│ ├── attacked/ # Attacked images
│ ├── extracted/ # Extracted watermarks
│ └── results.csv # Experiment summary
│
├── requirements.txt
└── README.md
```

---

## 6. Configuration Management

All system parameters are defined in `config/config.yaml`, including:

- Watermark size  
- Bit plane index  
- Color channel selection (OpenCV BGR format)  
- Attack parameters  
- Evaluation thresholds  

This approach ensures:
- Consistency across modules
- Reproducibility of experiments
- Safe integration of AI-generated code

---

## 7. AI-Assisted Development Strategy

Each module was developed with **explicit AI instructions** that include:

- Clear responsibility boundaries  
- Fixed function signatures  
- System-wide integration constraints  

By embedding integration rules directly into AI prompts, the project avoids:
- Logic inconsistency across modules  
- Platform-dependent behavior  
- Integration conflicts during final assembly  

This strategy enables **independent development on different devices**, followed by seamless integration.

---

## 8. Experimental Workflow

The complete experimental pipeline is as follows:

1. Prepare cover images and binary watermark  
2. Encode watermark into a bitstream  
3. Embed watermark into cover image using XOR bit-plane embedding  
4. Compute PSNR to evaluate imperceptibility  
5. Apply attacks:
   - JPEG compression (various quality factors)
   - Gaussian noise (different noise levels)
6. Extract watermark from attacked images  
7. Compute BER to evaluate robustness  
8. Save all results and summary tables  

---

## 9. How to Run

### 9.1 Install Dependencies

```bash
pip install -r requirements.txt
```

###9.2 Run the Full Experiment

From the project root directory:
```bash
python src/main.py
```

The script will automatically:

- Process all images in data/cover/
- Embed and extract watermarks
- Simulate attacks
- Generate output/results.csv

## 10. Experimental Results

Results are summarized in output/results.csv, including:

- Cover image name
- Attack type and parameters
- PSNR (dB)
- BER

Lower BER and higher PSNR indicate better performance.

## 11. Applications

Potential applications of this system include:

- Digital Rights Management (DRM)
- Image authentication
- Content traceability
- Ownership verification for online media

## 12. Conclusion

This project demonstrates not only a functional digital image watermarking system, but also a scalable and AI-friendly collaborative development methodology.

By combining classical image processing techniques with modern software engineering practices, the system achieves:

- Clear modularity
- Reproducible experiments
- Robust integration across devices and contributors

## 13. Notes
- Images are processed using OpenCV and follow the BGR color convention
- The system currently implements non-blind watermarking
- Future extensions may include blind extraction or frequency-domain watermarking
