import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def build_das_guide_docx(output_dir: str, docx_path: str = None, parameters=None) -> str:
    """
    Builds a professional, fully styled Word (.docx) document compatible with Google Docs,
    replicating the exact structure of instrucciones.pdf including all generated figures.
    """
    if docx_path is None:
        docx_path = os.path.join(output_dir, "RESULTADOS_DAS.docx")

    doc = Document()

    # Page setup: Normal margins (1 inch = 72 pt)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Styles helper
    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(18)
        run.font.color.rgb = RGBColor(20, 40, 80)
        doc.add_paragraph()

    def add_heading_1(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(30, 60, 110)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)

    def add_heading_2(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(40, 80, 140)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)

    def add_body(text):
        p = doc.add_paragraph(text)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        return p

    def add_image_if_exists(filename, width_inches=6.0, caption=""):
        fpath = os.path.join(output_dir, filename) if not os.path.isabs(filename) else filename
        if os.path.exists(fpath):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(6)
            p_img.paragraph_format.space_after = Pt(2)
            run = p_img.add_run()
            run.add_picture(fpath, width=Inches(width_inches))

            if caption:
                p_cap = doc.add_paragraph()
                p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_cap.paragraph_format.space_before = Pt(0)
                p_cap.paragraph_format.space_after = Pt(10)
                run_cap = p_cap.add_run(caption)
                run_cap.italic = True
                run_cap.font.size = Pt(9.5)
                run_cap.font.color.rgb = RGBColor(80, 80, 80)
        else:
            add_body(f"[Imagen no encontrada: {filename}]")

    # Document Header
    add_title("PROCESAMIENTO E INTERFEROMETRÍA DE DATOS DAS")
    add_body(
        "Recopilación de la metodología de procesamiento digital de señales, identificación de espectros de dispersión, "
        "reconstrucción interferométrica de funciones de Green empíricas y migración de ondas superficiales a partir de registros "
        "de fibra óptica (DAS)."
    )

    # 1. Control de calidad y preprocesamiento
    add_heading_1("1. Control de calidad y preprocesamiento:")
    add_heading_2("1.1 Datos brutos vs. Preprocesados")
    add_body(
        "Comparación de la tasa de deformación cruda (Strain Rate) registrada a lo largo del cable frente a los datos después de "
        "decimación temporal, filtro pasabanda Butterworth (0.8 - 4.5 Hz), normalización de 1-bit y blanqueamiento espectral."
    )
    add_image_if_exists(
        "1_das_qc_preprocessing.png",
        width_inches=6.2,
        caption="Figura 1.1: Cascada espacio-temporal de datos brutos (izquierda) y datos preprocesados normalizados (derecha)."
    )

    doc.add_page_break()

    # 2. Identificación de espectros
    add_heading_1("2. Identificación de espectros:")

    add_heading_2("2.1 Análisis frecuencia-número de onda (f-k)")
    add_body(
        "Espectro de dispersión f-k exhibiendo la cresta de energía asociada a las ondas superficiales de Rayleigh. "
        "Las líneas discontinuas indican las velocidades aparentes de referencia (1 km/s y 4 km/s)."
    )
    add_image_if_exists(
        "2_1_fk_dispersion_c.png",
        width_inches=5.2,
        caption="Figura 2.1: c) Espectro de dispersión f-k con número de onda k (1/km) vs frecuencia f (Hz) y pendientes de velocidad."
    )
    add_body(
        "A continuación se presentan los espectros f-k individuales calculados para cada uno de los archivos HDF5 analizados antes del apilado, organizados en grupos de 3 para una visualización amplia y detallada:"
    )
    for b in range(1, 4):
        b_name = f"6_fk_batch{b}.png"
        start_f = (b - 1) * 3 + 1
        end_f = b * 3
        add_image_if_exists(
            b_name,
            width_inches=6.2,
            caption=f"Figura 2.1.{b}: Espectros de potencia F-K individuales para los archivos {start_f} a {end_f}."
        )

    add_heading_2("2.2 Espectros de frecuencia-velocidad (f-v) y modelos 1D en profundidad")
    add_body(
        "Transformación al dominio frecuencia-velocidad de fase para resolver las ramas de modos normales y modos de fuga (leaky modes). "
        "A la derecha se presenta el perfil unidimensional de velocidad de onda de corte (Vs) en función de la profundidad (z)."
    )
    add_image_if_exists(
        "2_2_fv_and_1d_profile.png",
        width_inches=6.2,
        caption="Figura 2.2: Espectro de frecuencia-velocidad de fase (f-v) y b) modelo 1D de velocidad Vs en profundidad."
    )

    add_heading_2("2.3 Espectros f-s (frecuencia-lentitud)")
    add_body(
        "Representación de la dispersión en el dominio de lentitud de fase (Phase Slowness s = 1/v) en función de la frecuencia, "
        "con líneas de referencia a 0.5, 1.0 y 1.5 s/km."
    )
    add_image_if_exists(
        "2_3_fs_spectrum.png",
        width_inches=5.2,
        caption="Figura 2.3: Espectro frecuencia-lentitud de fase f-s."
    )

    doc.add_page_break()

    # 3. Correlación cruzada funciones de Green
    add_heading_1("3. Correlación cruzada funciones de Green:")
    add_heading_2("3.1 Funciones de Green empíricas (componentes causal y acausal) y forma de onda")
    add_body(
        "Resultados de la interferometría sísmica por correlación cruzada de ruido ambiental y su derivada temporal dC/dt. "
        "Se presenta el rango simétrico de retardos de -T a +T mostrando la componente acausal (t < 0) y causal (t > 0). "
        "A la derecha (a la par) se incluye la forma de onda recuperada para un offset representativo."
    )
    add_image_if_exists(
        "5_ccf_green_src0.png",
        width_inches=6.2,
        caption="Figura 3.1: Correlación cruzada acumulada C(t), Función de Green empírica dC/dt (acausal | causal) y forma de onda 1D a la par."
    )

    add_heading_2("3.2 Disparo virtual común apilado (Virtual Common Shot Gather)")
    add_body(
        "Disparo virtual extraído tras filtrado direccional F-K para aislar las ondas superficiales transmitidas hacia la red receptora."
    )
    add_image_if_exists(
        "2_virtual_shot_gather.png",
        width_inches=5.5,
        caption="Figura 3.2: Registro de disparo virtual apilado para la fuente virtual principal."
    )

    # 4. Migración RTM
    add_heading_1("4. Migración de ondas en reversa en el tiempo (TSW-RTM):")

    if parameters:
        add_heading_2("Parámetros utilizados")
        add_body(
            "Valores asignados que influyen directamente en el resultado de la migración. "
            "Cambiarlos modifica las imágenes de esta sección."
        )
        table = doc.add_table(rows=1, cols=3)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, text in zip(table.rows[0].cells, ("Bloque", "Parámetro", "Valor")):
            cell.text = ""
            run = cell.paragraphs[0].add_run(text)
            run.bold = True
            run.font.size = Pt(10)
        for group, name, value in parameters:
            cells = table.add_row().cells
            for cell, text in zip(cells, (group, name, value)):
                cell.text = ""
                cell.paragraphs[0].add_run(text).font.size = Pt(10)
        doc.add_paragraph()

    add_heading_2("4.1 Modelos de velocidad de inundación dual (Dual Flood Models)")
    add_body(
        "Generación de los modelos inundados de sedimento (vs, propagación hacia adelante) y roca metamórfica (vg, retropropagación). "
        "El modelo de velocidad de partida es un modelo supuesto, construido a partir de los parámetros indicados en la tabla anterior; "
        "no fue estimado a partir de los datos procesados en las secciones 1 a 3."
    )
    add_image_if_exists(
        "3_das_flood_models.png",
        width_inches=5.8,
        caption=(
            "Figura 4.1: Modelo de velocidad supuesto (a), definido con los parámetros de la tabla de parámetros utilizados "
            "y no obtenido de los datos, junto con los modelos de inundación de sedimento (b) y de roca (c) derivados de él. "
            "La falla se ubica de forma supuesta en el centro de la ventana."
        )
    )

    add_heading_2("4.2 Imagen migrada final de la falla tectónica")
    add_body(
        "Resultado final de la correlación cruzada de tiempo cero y filtrado Laplaciano 2D, enfocado en el plano de la discontinuidad."
    )
    add_image_if_exists(
        "4_das_migrated_fault_image.png",
        width_inches=5.8,
        caption="Figura 4.2: Imagen migrada TSW-RTM superpuesta sobre el modelo de velocidad de referencia."
    )

    doc.save(docx_path)
    print(f"Generated Word document: {docx_path}")
    return docx_path
