"""
Code adapted from https://github.com/ckjellson/textalloc
"""

from textalloc.non_overlapping_boxes import get_non_overlapping_boxes
import numpy as np
from typing import List, Tuple, Union


def allocate_text(
        x,
        y,
        text_list,
        fig,
        x_lims,
        y_lims,
        x_per_pixel,
        y_per_pixel,
        font,
        text_size,
        x_scatter: Union[np.ndarray, List[float]] = None,
        y_scatter: Union[np.ndarray, List[float]] = None,
        margin: float = 0.00,
        min_distance: float = 0.0075,
        max_distance: float = 0.07,
        draw_lines: bool = True,
        linecolor: str = "grey",
        draw_all: bool = True,
        nbr_candidates: int = 100,
):
    """
    Args:
        x: a list of x coordinates for the labels.
        y: a list of y coordinates for the labels.
        text_list:  a list of strings for the labels.
        fig (_type_): plotly dash figure.
        x_lims: the x limits of the plot.
        y_lims: the y limits of the plot.
        x_per_pixel: the x range per pixel.
        y_per_pixel: the y range per pixel.
        font (_type_): a pillow ImageFont object
        text_size (int): size of text.
        x_scatter (Union[np.ndarray, List[float]], optional): x-coords of all scattered points in plot 1d array/list.
        y_scatter (Union[np.ndarray, List[float]], optional): y-coords of all scattered points in plot 1d array/list.
        text_size (int): size of text.
        margin (float, optional): parameter for margins between objects. Increase for larger margins to points and lines.
        min_distance (float, optional): parameter for min distance between text and origin.
        max_distance (float, optional): parameter for max distance between text and origin.
        draw_lines (bool, optional): draws lines from original points to text-boxes.
        linecolor (str, optional): color code of the lines between points and text-boxes.
        draw_all (bool, optional): Draws all texts after allocating as many as possible despite overlap.
        nbr_candidates (int, optional): Sets the number of candidates used.
    """
    # Ensure good inputs
    x = np.array(x)
    y = np.array(y)
    assert len(x) == len(y)

    if x_scatter is not None:
        assert y_scatter is not None
    if y_scatter is not None:
        assert x_scatter is not None
        assert len(y_scatter) == len(x_scatter)
        x_scatter = x_scatter
        y_scatter = y_scatter
    assert min_distance <= max_distance
    assert min_distance >= margin

    # Create boxes in original plot
    original_boxes = []

    for x_coord, y_coord, s in zip(x, y, text_list):
        w, h = font.getlength(s) * x_per_pixel, text_size * y_per_pixel
        original_boxes.append((x_coord, y_coord, w, h, s))

    # Process extracted textboxes
    if x_scatter is None:
        scatterxy = None
    else:
        scatterxy = np.transpose(np.vstack([x_scatter, y_scatter]))
    non_overlapping_boxes, overlapping_boxes_inds = get_non_overlapping_boxes(
        original_boxes,
        x_lims,
        y_lims,
        margin,
        min_distance,
        max_distance,
        nbr_candidates,
        draw_all,
        scatter_xy=scatterxy,
    )

    if draw_lines:
        for x_coord, y_coord, w, h, s, ind in non_overlapping_boxes:
            x_near, y_near = find_nearest_point_on_box(
                x_coord, y_coord, w, h, x[ind], y[ind]
            )
            if x_near is not None:
                fig.add_annotation(
                    dict(
                        x=x[ind],
                        y=y[ind],
                        ax=x_near,
                        ay=y_near,
                        showarrow=True,
                        arrowcolor=linecolor,
                        text="",
                        axref='x',
                        ayref='y'

                    )
                )
    for x_coord, y_coord, w, h, s, ind in non_overlapping_boxes:
        fig.add_annotation(
            dict(
                x=x_coord,
                y=y_coord,
                showarrow=False,
                text=s,
                font=dict(size=text_size),
                xshift=w / (2 * x_per_pixel),
                yshift=h / (2 * y_per_pixel),
            )
        )

    if draw_all:
        for ind in overlapping_boxes_inds:
            fig.add_annotation(
                dict(
                    x=x[ind],
                    y=y[ind],
                    showarrow=False,
                    text=text_list[ind],
                    font=dict(size=text_size)
                )
            )


def get_annotation_data(
        x,
        y,
        text_list,
        x_lims,
        y_lims,
        x_per_pixel,
        y_per_pixel,
        font,
        x_scatter: Union[np.ndarray, List[float]] = None,
        y_scatter: Union[np.ndarray, List[float]] = None,
        text_size: int = 10,
        margin: float = 0.00,
        min_distance: float = 0.0075,
        max_distance: float = 0.07,
        draw_lines: bool = True,
        draw_all: bool = True,
        nbr_candidates: int = 100,
):
    """Main function of allocating text-boxes in matplotlib plot
    Args:
        x: a list of x coordinates for the labels.
        y: a list of y coordinates for the labels.
        text_list:  a list of strings for the labels.
        x_lims: the x limits of the plot.
        y_lims: the y limits of the plot.
        x_per_pixel: the x range per pixel.
        y_per_pixel: the y range per pixel.
        font: a pillow font object, used to obtain the width of the strings.
        x_scatter (Union[np.ndarray, List[float]], optional): x-coordinates of all scattered points in plot 1d array/list. Defaults to None.
        y_scatter (Union[np.ndarray, List[float]], optional): y-coordinates of all scattered points in plot 1d array/list. Defaults to None.
        text_size (int, optional): size of text. Defaults to 10.
        margin (float, optional): parameter for margins between objects. Increase for larger margins to points and lines. Defaults to 0.01.
        min_distance (float, optional): parameter for min distance between text and origin. Defaults to 0.015.
        max_distance (float, optional): parameter for max distance between text and origin. Defaults to 0.07.
        draw_all (bool, optional): Draws all texts after allocating as many as possible despit overlap. Defaults to True.
        nbr_candidates (int, optional): Sets the number of candidates used. Defaults to 0.
    """
    # Ensure good inputs
    x = np.array(x)
    y = np.array(y)
    assert len(x) == len(y)

    if x_scatter is not None:
        assert y_scatter is not None
    if y_scatter is not None:
        assert x_scatter is not None
        assert len(y_scatter) == len(x_scatter)
        x_scatter = x_scatter
        y_scatter = y_scatter
    assert min_distance <= max_distance
    assert min_distance >= margin

    # Create boxes in original plot
    original_boxes = []

    for x_coord, y_coord, string in zip(x, y, text_list):
        w, h = font.getlength(string) * x_per_pixel, text_size * y_per_pixel
        original_boxes.append((x_coord, y_coord, w, h, string))

    # Process extracted textboxes
    if x_scatter is None:
        scatterxy = None
    else:
        scatterxy = np.transpose(np.vstack([x_scatter, y_scatter]))
    non_overlapping_boxes, overlapping_boxes_inds = get_non_overlapping_boxes(
        original_boxes,
        x_lims,
        y_lims,
        margin,
        min_distance,
        max_distance,
        nbr_candidates,
        draw_all,
        scatter_xy=scatterxy,
    )

    lines = set()
    if draw_lines:
        for x_coord, y_coord, w, h, string, ind in non_overlapping_boxes:
            x_near, y_near = find_nearest_point_on_box(
                x_coord, y_coord, w, h, x[ind], y[ind]
            )
            if x_near is not None:
                lines.add((round(x[ind], 3), round(y[ind], 3), round(x_near, 3), round(y_near, 3)))

    annotations = set()
    for x_coord, y_coord, width, height, string, ind in non_overlapping_boxes:
        annotations.add((round(x_coord, 3), round(y_coord, 3), string, round(width, 3), round(height, 3)))
    return annotations, lines


def find_nearest_point_on_box(
        xmin: float, ymin: float, w: float, h: float, x: float, y: float
) -> Tuple[float, float]:
    """Finds nearest point on box from point.
    Returns None,None if point inside box
    Args:
        xmin (float): xmin of box
        ymin (float): ymin of box
        w (float): width of box
        h (float): height of box
        x (float): x-coordinate of point
        y (float): y-coordinate of point
    Returns:
        Tuple[float, float]: x,y coordinate of nearest point
    """
    xmax = xmin + w
    ymax = ymin + h
    if x < xmin:
        if y < ymin:
            return xmin, ymin
        elif y > ymax:
            return xmin, ymax
        else:
            return xmin, y
    elif x > xmax:
        if y < ymin:
            return xmax, ymin
        elif y > ymax:
            return xmax, ymax
        else:
            return xmax, y
    else:
        if y < ymin:
            return x, ymin
        elif y > ymax:
            return x, ymax
    return None, None


def lines_to_segments(
        x_lines: List[np.ndarray],
        y_lines: List[np.ndarray],
) -> np.ndarray:
    """Sets up
    Args:
        x_lines (List[np.ndarray]): x-coordinates of all lines in plot list of 1d arrays
        y_lines (List[np.ndarray]): y-coordinates of all lines in plot list of 1d arrays
    Returns:
        np.ndarray: 2d array of line segments
    """
    assert len(x_lines) == len(y_lines)
    n_x_segments = np.sum([len(line_x) - 1 for line_x in x_lines])
    n_y_segments = np.sum([len(line_y) - 1 for line_y in y_lines])
    assert n_x_segments == n_y_segments
    lines_xyxy = np.zeros((n_x_segments, 4))
    iter = 0
    for line_x, line_y in zip(x_lines, y_lines):
        for i in range(len(line_x) - 1):
            lines_xyxy[iter, :] = [line_x[i], line_y[i], line_x[i + 1], line_y[i + 1]]
            iter += 1
    return lines_xyxy
