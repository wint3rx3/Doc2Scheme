import cv2
import numpy as np
import pandas as pd
import easyocr
import torch

def extract_text_from_cells(cells_data): 
    """
    셀 데이터에서 텍스트를 추출합니다.
    
    :param cells_data: 셀 정보 리스트
    :return: 셀 내 텍스트들을 이어붙인 문자열
    """
    extracted_text = []
    for cell in cells_data:
        if 'text' in cell:
            extracted_text.append(cell['text'])
    return ' '.join(extracted_text)

class TableExtractor:
    def __init__(self):
        self.reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
    
    def process_image(self, image):
        """
        이미지에서 표 영역의 선을 검출하고 셀별 텍스트 및 그리드 정보를 추출합니다.
        
        :param image: 표 영역 이미지 (numpy array)
        :return: 표 결과 딕셔너리 (셀 정보, 그리드 정보 포함)
        """
        if isinstance(image, str):
            self.image = cv2.imread(image)
        else:
            self.image = image.copy()
        self.result = self.image.copy()
        self.detect_lines()
        self.classify_lines_and_find_intersections()
        self.remove_duplicate_points()
        data, extracted_cells = self.extract_text_from_cells()
        
        # DataFrame으로 후처리
        df = pd.DataFrame(data)
        df = df.replace(r'^\s*$', np.nan, regex=True).dropna(how='all', axis=0).dropna(how='all', axis=1)
        df = df.reset_index(drop=True).fillna('')
        processed_cells = []
        for i in range(len(df)):
            for j in range(len(df.columns)):
                original_cell = next((cell for cell in extracted_cells if cell['row'] == i + 1 and cell['col'] == j + 1), None)
                if original_cell:
                    processed_cells.append({
                        'row': i + 1,
                        'col': j + 1,
                        'text': df.iloc[i, j],
                        'coordinates': original_cell['coordinates']
                    })
        final_result = {'cells': processed_cells, 'grid_info': {'rows': len(df), 'cols': len(df.columns)}}
        return final_result
    
    def detect_lines(self):
        self.edges = cv2.Canny(self.image, 50, 150, apertureSize=3)
        self.lines = cv2.HoughLinesP(self.edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    def classify_lines_and_find_intersections(self):
        self.intersection_points = []
        self.horizontal_lines = []
        self.vertical_lines = []
        if self.lines is not None:
            for line in self.lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                if angle < 10 or angle > 170:
                    self.horizontal_lines.append(line[0])
                elif 80 < angle < 100:
                    self.vertical_lines.append(line[0])
            height, width = self.image.shape[:2]
            margin = 10
            self.horizontal_lines.extend([[margin, margin, width - margin, margin],
                                          [margin, height - margin, width - margin, height - margin]])
            self.vertical_lines.extend([[margin, margin, margin, height - margin],
                                        [width - margin, margin, width - margin, height - margin]])
            self._find_intersection_points()
            self._process_end_points()
    
    def _find_intersection_points(self):
        for h_line in self.horizontal_lines:
            for v_line in self.vertical_lines:
                x1, y1, x2, y2 = h_line
                x3, y3, x4, y4 = v_line
                denominator = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
                if denominator != 0:
                    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
                    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        x = int(x1 + t * (x2 - x1))
                        y = int(y1 + t * (y2 - y1))
                        self.intersection_points.append((x, y))
        self.intersection_points = sorted(set(self.intersection_points), key=lambda p: (p[1], p[0]))
    
    def _process_end_points(self):
        end_points = []
        for line in self.horizontal_lines + self.vertical_lines:
            x1, y1, x2, y2 = line
            end_points.extend([(x1, y1), (x2, y2)])
        x_values = [pt[0] for pt in end_points]
        y_values = [pt[1] for pt in end_points]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        self.filtered_end_points = [(x, y) for (x, y) in end_points if (x_min <= x <= x_min + 10 or x_max - 10 <= x <= x_max) or (y_min <= y <= y_min + 10 or y_max - 10 <= y <= y_max)]
        self.all_points = self.intersection_points + self.filtered_end_points
    
    def remove_duplicate_points(self, distance_threshold=15):
        self.unique_points = []
        for point in self.all_points:
            if all(np.linalg.norm(np.array(point) - np.array(unique_point)) > distance_threshold for unique_point in self.unique_points):
                self.unique_points.append(point)
    
    def extract_text_from_cells(self, min_height=30, min_width=30):
        self.x_coords = sorted(list(set([pt[0] for pt in self.intersection_points])))
        self.y_coords = sorted(list(set([pt[1] for pt in self.intersection_points])))
        data = []
        extracted_cells = []
        for i in range(len(self.y_coords) - 1):
            row = []
            for j in range(len(self.x_coords) - 1):
                top_left_x = self.x_coords[j]
                top_left_y = self.y_coords[i]
                bottom_right_x = self.x_coords[j + 1]
                bottom_right_y = self.y_coords[i + 1]
                tile = self.image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                cell_info = {'row': i + 1, 'col': j + 1,
                             'coordinates': {'top_left': (top_left_x, top_left_y),
                                             'bottom_right': (bottom_right_x, bottom_right_y)}}
                if tile.shape[0] < min_height or tile.shape[1] < min_width:
                    row.append("")
                    cell_info['text'] = ""
                    extracted_cells.append(cell_info)
                    continue
                text_result = self.reader.readtext(tile, detail=0)
                text = "\n".join(text_result).strip()
                row.append(text)
                cell_info['text'] = text
                extracted_cells.append(cell_info)
            data.append(row)
        return data, extracted_cells

def process_table_regions(table_regions):
    """
    표 영역 리스트에 대해 텍스트 및 셀 정보를 추출하고 최종 결과 리스트를 반환합니다.
    
    :param table_regions: 표 영역 리스트
    :return: 표 영역 처리 결과 리스트
    """
    table_extractor = TableExtractor()
    results = []
    for region in table_regions:
        unique_id = region["unique_id"]
        try:
            table_result = table_extractor.process_image(region["image"])
        except Exception as e:
            print(f"{unique_id}에서 표 추출 실패: {e}")
            continue
        table_text = extract_text_from_cells(table_result["cells"])
        results.append({
            "data_id": unique_id,
            "page_number": region["page_number"],
            "region_type": "일반표",
            "content": table_text,
            "meta": {
                "bounding_box": region["bounding_box"],
                "cells": table_result.get("cells", []),
                "grid": table_result.get("grid_info", {})
            }
        })
    return results
