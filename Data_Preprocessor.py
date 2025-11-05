import pandas as pd

class perProcesser:
    def init(self, filepath):
        self.filepath = filepath
        self.data = None

    def read_data(self):
        try:
            if self.filepath.lower().endswith(".csv"):
                self.data = pd.read_csv(self.filepath)
            elif self.filepath.lower().endswith((".xls", ".xlsx")):
                self.data = pd.read_excel(self.filepath)
            else:
                print("Unsupported file type")
                return None
            return self.data

            
    
        except Exception as e:
            print(f"Error reading file: {e}")
            return None