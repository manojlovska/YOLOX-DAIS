import os
from PIL import Image, ImageDraw
from pathlib import Path
from lxml import etree


class Visualizer:
    def __init__(self, image_path, width, height, square_size):
        self.image_path = image_path
        self.width = width
        self.height = height
        self.square_size = square_size

        self.x_squares = int(width / square_size)
        self.y_squares = int(height / square_size)

        self.im = Image.open(image_path)
        self.im = self.im.resize((width, height))
        self.draw = ImageDraw.Draw(self.im)

    # get and scale all polylines for an image
    def get_polylines(self, annotation_path):
        image_name = os.path.basename(self.image_path)
        # import pdb; pdb.set_trace()

        path_expression = "//image[contains(@name,'" + image_name + "')]"
        annotation_file = etree.parse(annotation_path)
        annotated_image = annotation_file.xpath(path_expression)[0]

        original_width = int(annotated_image.get("width"))
        original_height = int(annotated_image.get("height"))

        x_scale = self.width / original_width
        y_scale = self.height / original_height

        polylines = []

        poly_element = annotated_image.xpath("./polyline[@label='mag_tape']")
        for polyline in poly_element:
            parsed_polyline = [r.split(",") for r in [
                r for r in polyline.get("points").split(";")]]
            polylines += [parsed_polyline]

        # convert polylines to float and optionally scale them
        for polyline in polylines:
            for line in polyline:
                line[0] = float(line[0]) * x_scale
                line[1] = float(line[1]) * y_scale

        return polylines

    def draw_grid(self, color="black"):
        self.draw.line((0, 0, 0, self.height - 1), fill=0)
        self.draw.line((0, 0, self.width - 1, 0), fill=0)
        for i in range(1, int(self.width/self.square_size) + 1):
            self.draw.line(((i * self.square_size) - 1, 0, (i *
                           self.square_size) - 1, self.height - 1), fill=color)
        for i in range(0, int(self.height/self.square_size) + 1):
            self.draw.line((0, (i * self.square_size) - 1, self.width -
                           1, (i * self.square_size) - 1), fill=color)

    # draw ground truth polyline
    def draw_true_polylines(self, polylines, color="red"):
        for polyline in polylines:
            x1, y1 = polyline[0]
            for line in polyline[1:]:
                x2, y2 = line
                self.draw.line([x1, y1, x2, y2], fill=color)
                x1, y1 = line

    def save_image(self, out_path="./out/", name=""):
        if name == "":
            name = Path(self.image_path).stem

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        self.im.save(os.path.join(out_path,name))

    def show_image(self):
        self.im.show()

    def draw_cartesian_predictors(self, squares, thresh=0.5, color="green"):
        for i in range(0, self.x_squares):
            for j in range(0, self.y_squares):
                if squares[i][j][4] < thresh:
                    continue

                # coordinates of the top left corner of each square
                x1 = (i * self.square_size)
                y1 = (j * self.square_size)

                if x1 != 0:
                    x1 -= 1
                if y1 != 0:
                    y1 -= 1

                x2 = x1
                y2 = y1

                x1 += int(squares[i][j][0] * self.square_size)
                y1 += int((1 - squares[i][j][1]) * self.square_size)

                x2 += int(squares[i][j][2] * self.square_size)
                y2 += int((1 - squares[i][j][3]) * self.square_size)

                self.draw.line([(x1, y1), (x2, y2)], width=4, fill=color)

    def draw_cartesian_intersections(self, squares, thresh=0.5, color="yellow"):
        for i in range(0, self.x_squares):
            for j in range(0, self.y_squares):
                if squares[i][j][4] < thresh:
                    continue

                # coordinates of the top left corner of each square
                x1 = (i * self.square_size)
                y1 = (j * self.square_size)

                if x1 != 0:
                    x1 -= 1
                if y1 != 0:
                    y1 -= 1

                x2 = x1
                y2 = y1

                x1 += int(squares[i][j][0] * self.square_size)
                y1 += int((1 - squares[i][j][1]) * self.square_size)

                x2 += int(squares[i][j][2] * self.square_size)
                y2 += int((1 - squares[i][j][3]) * self.square_size)

                self.draw.line([(x1, y1), (x1, y1)], fill=color)
                self.draw.line([(x2, y2), (x2, y2)], fill=color)
