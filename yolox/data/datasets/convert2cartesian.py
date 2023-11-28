import numpy as np
from shapely.geometry import LineString
from math import hypot

class Converter:
    def __init__(self, width, height, square_size):
        self.height = height
        self.width = width
        self.square_size = square_size

        self.x_squares = int(width / square_size)
        self.y_squares = int(height / square_size)

        # extrapolate a line if it's this complete allready (percantage)
        self.extrapolation_threshold = 0.5
        # accounts for annotation mistakes on edges (in pixels)
        self.edge_margin = 2
        # if distance between two cartesian points is greater than this, stop looking for more points (cartesian)
        self.min_cart_distance = 0.02
        # if the longest upper line in a square is this times longer than the longest bottom one, pick it (percentage)
        self.upper_line_threshold = 2.5

    
    # first line is annotaded line
    def get_interstection(self, first_line, second_line, width, height):
        # account for annnotation mistakes
        #   for x on the left
        if first_line[0][0] - self.edge_margin <= 0:
            first_line[0][0] = 0
        if first_line[1][0] - self.edge_margin <= 0:
            first_line[1][0] = 0
        #   for x on the right
        if first_line[0][0] + self.edge_margin >= width:
            first_line[0][0] = width - 1
        if first_line[1][0] + self.edge_margin >= width:
            first_line[1][0] = width - 1
        #   for y on the top
        if first_line[0][1] - self.edge_margin <= 0:
            first_line[0][1] = 0
        if first_line[1][1] - self.edge_margin <= 0:
            first_line[1][1] = 0
        #   for y on the bottomt
        if first_line[0][1] + self.edge_margin >= height:
            first_line[0][1] = width - 1
        if first_line[1][1] + self.edge_margin >= height:
            first_line[1][1] = width - 1

        line1 = LineString(first_line)
        line2 = LineString(second_line)

        if not line1.intersects(line2):
            return False

        int_pt = line1.intersection(line2)
        if int_pt and hasattr(int_pt, "x") and hasattr(int_pt, "y"):
            return [int(int_pt.x), int(int_pt.y)]

        return False
    
    def cartesian_line(self, points):
        p1 = points[0]
        p2 = points[1]

        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])

        return A, B, -C
    
    def get_cartesian_intersection(self, line1, line2):
        L1 = self.cartesian_line(line1)
        L2 = self.cartesian_line(line2)

        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]

        if D != 0:
            x = Dx / D
            y = Dy / D
            if x < 0 or x > 1 or y < 0 or y > 1:
                return False
            return x, y
        else:
            return False
        
    # returns all squares that touch the given point
    def get_touching_squares(self, point, width, height, square_size):
        x = int(point[0])
        y = int(point[1])

        wanted_squares = []

        initial_square = [int(x / square_size), int(y / square_size)]
        wanted_squares += [initial_square]

        if (x + 1) % square_size == 0 and x != width - 1:
            wanted_squares += [[initial_square[0] + 1, initial_square[1]]]

        if (y + 1) % square_size == 0 and y != height - 1:
            wanted_squares += [[initial_square[0], initial_square[1] + 1]]

        if (x + 1) % square_size == 0 and (y + 1) % square_size == 0 and x != width - 1 and y != height - 1:
            wanted_squares += [[initial_square[0] + 1, initial_square[1] + 1]]

        return wanted_squares
    
    # convert x, y coordiante on the image to cartesian coordinate for a specific square
    def to_cartesian_xy(self, square_indexes, point, square_size):
        x = point[0]
        y = point[1]

        # coordinates of the top left corner of each square
        x1 = (square_indexes[0] * square_size)
        y1 = (square_indexes[1] * square_size)

        if x1 != 0:
            x1 -= 1
        if y1 != 0:
            y1 -= 1

        new_x = (x - x1) / square_size
        new_y = 1 - ((y - y1) / square_size)

        return [new_x, new_y]
    
    # adds cartesian intersected point to the wanted square
    def add_intersected_point(self, squares, polyline, square_index, cartesian_point):
        square_index = str(square_index)
        if squares[polyline]:
            if square_index in squares[polyline]:
                squares[polyline][square_index] += [cartesian_point]
            else:
                squares[polyline][square_index] = [cartesian_point]
        else:
            squares[polyline] = {square_index: [cartesian_point]}

    def line_lenght(self, points):
        x1 = points[0][0]
        y1 = points[0][1]
        x2 = points[1][0]
        y2 = points[1][1]

        return hypot(x1 - x2, y1 - y2)
    
    # get the longest line in a square that allings with given conditions
    def get_longest_cond_line(self, intersections, top_to_bottom=False):
        intersections.sort(key=lambda x: x[1], reverse=top_to_bottom)
        interesting_intersections = []
        for intersection in intersections:
            interesting_intersections += [intersection]
            first_intersection = interesting_intersections[0]
            if len(interesting_intersections) > 1 and (abs(intersection[1] - first_intersection[1]) >= self.min_cart_distance
                                                    and abs(intersection[0] - first_intersection[0]) >= self.min_cart_distance):
                break

        longest_line = [interesting_intersections[0], interesting_intersections[1]]
        for intersection1 in interesting_intersections:
            for intersection2 in interesting_intersections:
                if intersection1 == intersection2 or [intersection2, intersection1] in longest_line:
                    continue
                if (self.line_lenght(longest_line) < self.line_lenght([intersection1, intersection2])):
                    longest_line = [intersection1, intersection2]

        return longest_line

    def to_cartesian(self, polylines):
        # [[[x1, y1, x2, y2, p]]]
        squares_output = np.empty((self.x_squares, self.y_squares, 5), object)

        # get intersections
        intersections = []
        intersected_squares = np.empty((len(polylines)), object)
        for polyline_index, polyline in enumerate(polylines):
            # get intersections of each polyline seperatly
            intersections = []
            x1, y1 = polyline[0]

            for line in polyline[1:]:
                x2, y2 = line
                first_line = [[x1, y1], [x2, y2]]

                # get intersections of the vertical lines
                #   left most line
                #   TODO skip if wanted line doesn't touch the left
                second_line = [(0, 0), (0, self.height - 1)]
                intersection = self.get_interstection(
                    first_line, second_line, self.width, self.height)
                if intersection and (intersection not in intersections):
                    intersections += [intersection]
                # all other lines
                # TODO go only through the vertical lines between wanted line coordinates
                for i in range(1, int(self.width/self.square_size) + 1):
                    second_line = [((i * self.square_size) - 1, 0),
                                   ((i * self.square_size) - 1, self.height - 1)]
                    intersection = self.get_interstection(
                        first_line, second_line, self.width, self.height)

                    if intersection and intersection not in intersections:
                        intersections += [intersection]

                # get intersections of the horizontal lines
                #   top most line
                #   TODO skip if wanted line doesn't touch the top
                second_line = [(0, 0), (self.width - 1, 0)]
                intersection = self.get_interstection(
                    first_line, second_line, self.width, self.height)
                if intersection and intersection not in intersections:
                    intersections += [intersection]
                # all other lines
                # TODO go only through the horizontal lines between wanted line coordinates
                for i in range(1, int(self.height/self.square_size) + 1):
                    second_line = [(0, (i * self.square_size) - 1),
                                   (self.width - 1, (i * self.square_size) - 1)]
                    intersection = self.get_interstection(
                        first_line, second_line, self.width, self.height)

                    if intersection and intersection not in intersections:
                        intersections += [intersection]
                x1, y1 = line

            # add wanted squares of this polyline to intersected_squares
            for intersection in intersections:
                for square_index in self.get_touching_squares(intersection, self.width, self.height, self.square_size):
                    self.add_intersected_point(intersected_squares, polyline_index, square_index, self.to_cartesian_xy(
                        square_index, intersection, self.square_size))

        # if a square has more than two intersections for a polyline, choose only two of them
        for i in range(0, len(intersected_squares)):
            if not intersected_squares[i]:
                continue
            for key in list(intersected_squares[i].keys()):
                intersections = intersected_squares[i][key]
                if len(intersections) <= 2:
                    continue

                # we assume the square can have up to two intesections
                longest_line_bt = self.get_longest_cond_line(intersections)
                longest_line_tb = self.get_longest_cond_line(
                    intersections, top_to_bottom=True)

                if self.line_lenght(longest_line_tb) / self.line_lenght(longest_line_bt) > self.upper_line_threshold:
                    intersected_squares[i][key] = longest_line_tb
                else:
                    intersected_squares[i][key] = longest_line_bt

        # extrapolate a polyline line if it's long enough within a square
        for polyline_index, polyline in enumerate(polylines):
            for point in [polyline[0], polyline[len(polyline) - 1]]:
                touching_squares = []
                for square in self.get_touching_squares(point, self.width, self.height, self.square_size):
                    if intersected_squares[polyline_index] and str(square) in intersected_squares[polyline_index]\
                            and len(intersected_squares[polyline_index][str(square)]) == 1:
                        touching_squares += [square]
                if len(touching_squares) != 1:
                    continue
                square_index = str(touching_squares[0])

                # check if the square allready has a line going through it
                square_occupied = False
                for dict in intersected_squares:
                    if dict and square_index in dict and len(dict[square_index]) > 1:
                        square_occupied = True
                        break
                if square_occupied:
                    continue

                # extrapolate
                established_point = intersected_squares[polyline_index][square_index][0]
                cartesian_poly_point = self.to_cartesian_xy(
                    touching_squares[0], point, self.square_size)
                longest_line_length = 0
                longest_line = []

                cartesian_edges = [[[0, 0], [0, 1]], [
                    [0, 0], [1, 0]], [[0, 1], [1, 1]], [[1, 1], [1, 0]]]
                for edge in cartesian_edges:
                    edge_intersection = self.get_cartesian_intersection(
                        [cartesian_poly_point, established_point], edge)
                    if edge_intersection and self.line_lenght([established_point, edge_intersection]) > longest_line_length:
                        longest_line_length = self.line_lenght(
                            [established_point, edge_intersection])
                        longest_line = [established_point, edge_intersection]

                if longest_line_length and self.line_lenght([established_point, cartesian_poly_point]) / longest_line_length >= self.extrapolation_threshold:
                    intersected_squares[polyline_index][square_index] = longest_line

        # remove squares that have only one intersecting point for a polyline
        for i in range(0, len(intersected_squares)):
            if not intersected_squares[i]:
                continue
            for key in list(intersected_squares[i].keys()):
                if len(intersected_squares[i][key]) == 1:
                    del intersected_squares[i][key]

        # if more than one polyline goes through a square, leave only the polyline with the longest intersection line
        for i in range(0, len(intersected_squares)):
            if not intersected_squares[i]:
                continue
            for key in list(intersected_squares[i].keys()):
                for j in range(0, len(intersected_squares)):
                    if i == j or (not intersected_squares[j]) or not key in intersected_squares[j]:
                        continue
                    if self.line_lenght(intersected_squares[i][key]) <= self.line_lenght(intersected_squares[j][key]):
                        del intersected_squares[i][key]

        # add all squares from intersected_squares to the final output squares_output
        for i in range(0, len(intersected_squares)):
            if not intersected_squares[i]:
                continue
            for key in intersected_squares[i].keys():
                square_index = key.strip('][').split(', ')
                n = int(square_index[0])
                j = int(square_index[1])

                temp = list(
                    intersected_squares[i][key][0]) + list(intersected_squares[i][key][1]) + [1]
                squares_output[n][j] = temp

        return squares_output.astype(np.float32)
