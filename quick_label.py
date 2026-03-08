class Predict:
    
    def __init__(self, tag_detection_model: str, row_detection_model: str, ocr_model: str):
        
        self.tag_detection_model = tag_detection_model
        self.row_detection_model = row_detection_model
        self.ocr_model = ocr_model
        
    def slice_frame(self, frame):
        pass
        
    def run_prediction_pipeline(self):
        pass
    
    def predict_tags(self, frame):
        pass
    
    def predict_last_row(self, frame):
        pass
    
    def predict_cow_id(self, frame):
        pass
        