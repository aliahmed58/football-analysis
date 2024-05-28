# import json
# import requests
# from io import BytesIO
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib.units import inch
# from PIL import Image
# import tempfile

# # Function to download image and save it to a temporary file
# def download_image(url):
#     response = requests.get(url)
#     img = Image.open(BytesIO(response.content))
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
#     img.save(temp_file.name)
#     return temp_file.name


# # Function to split long text into lines
# def split_text(text, max_length=40):
#     lines = text.split("\n")
#     split_lines = []
#     for line in lines:
#         words = line.split()
#         current_line = []
#         current_length = 0
#         for word in words:
#             if current_length + len(word) + 1 > max_length:
#                 split_lines.append(" ".join(current_line))
#                 current_line = [word]
#                 current_length = len(word) + 1
#             else:
#                 current_line.append(word)
#                 current_length += len(word) + 1
#         if current_line:
#             split_lines.append(" ".join(current_line))
#     return split_lines


# # Function to add the heading and logo
# def add_heading_and_logo(pdf, logo_url, heading_text):
#     logo_path = logo_url
#     img = Image.open(logo_path)
#     img_width, img_height = img.size
#     aspect = img_height / float(img_width)
#     img_width = 1.5 * inch
#     img_height = img_width * aspect
#     img_x = width - margin - img_width
#     img_y = height - margin - img_height
#     pdf.drawImage(logo_path, img_x, img_y, width=img_width, height=img_height)

#     pdf.setFont("Helvetica-Bold", 30)
#     heading_width = pdf.stringWidth(heading_text, "Helvetica-Bold", 20)
#     heading_x = (width - heading_width) / 3  # Centered horizontally
#     heading_y = height - margin - (img_height / 2)
#     pdf.drawString(heading_x, heading_y, heading_text)


# def add_section(title, section_data, y):
#     pdf.setFont("Helvetica-Bold", 16)
#     pdf.drawString(margin, y, title)
#     y -= 20

#     images = section_data["images"]
#     kpis = {k: v for k, v in section_data.items() if k != "images"}

#     # Draw Images and KPIs
#     img_y = y
#     for label, img_url in images.items():
#         img_path = download_image(img_url)
#         img = Image.open(img_path)
#         img_width, img_height = img.size
#         aspect = img_height / float(img_width)
#         img_width = 5 * inch  # Increased width
#         img_height = img_width * aspect
#         img_y -= img_height + 10
#         pdf.drawImage(img_path, margin, img_y, width=img_width, height=img_height)
#         pdf.setFont("Helvetica", 12)
#         kpi_text = get_kpi_text(label, kpis)
#         text_x = margin + img_width + 10
#         text_y = img_y + img_height - 12
#         for line in split_text(kpi_text):
#             pdf.drawString(text_x, text_y, line)
#             text_y -= 14

#         if img_y < margin + img_height:
#             pdf.showPage()
#             img_y = height - margin - 20


# def get_kpi_text(label, kpis):
#     if label == "passes_completed":
#         return f"Complete: {kpis['passes']['complete']}"
#     elif label == "passes_incomplete":
#         return f"Incomplete: {kpis['passes']['incomplete']}"
#     elif label == "receiving":
#         return f"X Max Pressure: {kpis['x_max_pressure']}\nY Max Pressure: {kpis['y_max_pressure']}"
#     elif label == "pressure":
#         return f"Avg Pressure: {kpis['avg_pressure']}\nMin Pressure: {kpis['min_pressure']}\nMax Pressure: {kpis['max_pressure']}"
#     elif label == "possession":
#         return f"Avg X: {kpis['avg_x']}\nAvg Y: {kpis['avg_y']}"
#     else:
#         return ""


# if __name__ == "__main__":
#     # Load JSON Data from File
#     with open("test.json", "r") as file:
#         data = json.load(file)

#     # print(data["Home"])

#     # Add heading and logo
#     heading_text = "Presented by FAN"
#     logo_path = (
#         "Designer (1).png"  # Replace with the actual local path to the logo image
#     )

#     # Create PDF
#     pdf_file = "output.pdf"
#     pdf = canvas.Canvas(pdf_file, pagesize=letter)
#     width, height = letter
#     margin = 30
#     add_heading_and_logo(pdf, logo_path, heading_text)

#     # Add vertical offset
#     vertical_offset = 150  # Adjust as needed
#     y = height - margin - vertical_offset
#     # Add sections for Home and Away
#     add_section("Home Team", data["Home"], y)
#     pdf.showPage()
#     y = height - margin - 50
#     add_section("Away Team", data["Away"], y)

#     # Save the PDF
#     pdf.save()

#     print(f"PDF generated: {pdf_file}")
