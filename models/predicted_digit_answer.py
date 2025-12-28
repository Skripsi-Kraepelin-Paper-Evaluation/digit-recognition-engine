class PredictedDigitAnswer:
    def __init__(self, digit=None, accuracy=None, column=None, row=None, need_manual_check=False, checked=False, is_blank=False, is_skipped=False):
        self.digit = digit
        self.accuracy = accuracy
        self.column = column
        self.row = row
        self.need_manual_check = need_manual_check
        self.checked = checked
        self.is_blank = is_blank

        # validation
        if row == None or column == None:
            print("row or column is null")
            exit()
        if (digit == None or accuracy== None) and is_blank == False:
            print("digit predicted or accuracy is null")
            exit()

    def serialize_obj(self):
        return {
            'digit': int(self.digit) if self.digit is not None else None,
            'accuracy': float(self.accuracy) if self.accuracy is not None else None,
            'column': int(self.column),
            'row': int(self.row),
            'need_manual_check': bool(self.need_manual_check),
            'checked': bool(self.checked),
            'is_blank': bool(self.is_blank),
        }