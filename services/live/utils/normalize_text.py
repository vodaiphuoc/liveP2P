import re

class VietnameseTTSNormalizer:
    """
    A text normalizer for Vietnamese Text-to-Speech systems.
    Converts numbers, dates, units, and special characters into readable Vietnamese text.
    """
    
    def __init__(self):
        self.units = {
            'km': 'ki lô mét', 'dm': 'đê xi mét', 'cm': 'xen ti mét',
            'mm': 'mi li mét', 'nm': 'na nô mét', 'µm': 'mic rô mét',
            'μm': 'mic rô mét', 'm': 'mét',
            
            'kg': 'ki lô gam', 'g': 'gam', 'mg': 'mi li gam',
            
            'km²': 'ki lô mét vuông', 'km2': 'ki lô mét vuông',
            'm²': 'mét vuông', 'm2': 'mét vuông',
            'cm²': 'xen ti mét vuông', 'cm2': 'xen ti mét vuông',
            'mm²': 'mi li mét vuông', 'mm2': 'mi li mét vuông',
            'ha': 'héc ta',
            
            'km³': 'ki lô mét khối', 'km3': 'ki lô mét khối',
            'm³': 'mét khối', 'm3': 'mét khối',
            'cm³': 'xen ti mét khối', 'cm3': 'xen ti mét khối',
            'mm³': 'mi li mét khối', 'mm3': 'mi li mét khối',
            'l': 'lít', 'dl': 'đê xi lít', 'ml': 'mi li lít', 'hl': 'héc tô lít',
            
            'v': 'vôn', 'kv': 'ki lô vôn', 'mv': 'mi li vôn',
            'a': 'am pe', 'ma': 'mi li am pe', 'ka': 'ki lô am pe',
            'w': 'oát', 'kw': 'ki lô oát', 'mw': 'mê ga oát', 'gw': 'gi ga oát',
            'kwh': 'ki lô oát giờ', 'mwh': 'mê ga oát giờ', 'wh': 'oát giờ',
            'ω': 'ôm', 'ohm': 'ôm', 'kω': 'ki lô ôm', 'mω': 'mê ga ôm',
            
            'hz': 'héc', 'khz': 'ki lô héc', 'mhz': 'mê ga héc', 'ghz': 'gi ga héc',
            
            'pa': 'pát cal', 'kpa': 'ki lô pát cal', 'mpa': 'mê ga pát cal',
            'bar': 'ba', 'mbar': 'mi li ba', 'atm': 'át mốt phia', 'psi': 'pi ét xai',
            
            'j': 'giun', 'kj': 'ki lô giun',
            'cal': 'ca lo', 'kcal': 'ki lô ca lo',
        }
        
        self.digits = ['không', 'một', 'hai', 'ba', 'bốn', 
                      'năm', 'sáu', 'bảy', 'tám', 'chín']
    
    def normalize(self, text):
        """Main normalization pipeline with EN tag protection."""
        # Step 1: Extract and protect EN tags
        en_contents = []
        placeholder_pattern = "___EN_PLACEHOLDER_{}___ "
        
        def extract_en(match):
            en_contents.append(match.group(0))
            return placeholder_pattern.format(len(en_contents) - 1)
        
        text = re.sub(r'<en>.*?</en>', extract_en, text, flags=re.IGNORECASE)
        
        # Step 2: Normal normalization pipeline
        text = text.lower()
        text = self._normalize_temperature(text)
        text = self._normalize_currency(text)
        text = self._normalize_percentage(text)
        text = self._normalize_units(text)
        text = self._normalize_time(text)
        text = self._normalize_date(text)
        text = self._normalize_phone(text)
        text = self._normalize_versions(text)
        text = self._normalize_numbers(text)
        text = self._number_to_words(text)
        text = self._normalize_special_chars(text)
        text = self._normalize_whitespace(text)
        
        # Step 3: Restore EN tags
        for idx, en_content in enumerate(en_contents):
            text = text.replace(placeholder_pattern.format(idx).lower(), en_content + ' ')
        
        # Final whitespace cleanup
        text = self._normalize_whitespace(text)
        
        return text
    
    def _normalize_temperature(self, text):
        """Convert temperature notation to words."""
        text = re.sub(r'-(\d+(?:[.,]\d+)?)\s*°\s*c\b', r'âm \1 độ xê', text, flags=re.IGNORECASE)
        text = re.sub(r'-(\d+(?:[.,]\d+)?)\s*°\s*f\b', r'âm \1 độ ép', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*°\s*c\b', r'\1 độ xê', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*°\s*f\b', r'\1 độ ép', text, flags=re.IGNORECASE)
        text = re.sub(r'°', ' độ ', text)
        return text
    
    def _normalize_currency(self, text):
        """Convert currency notation to words."""
        def decimal_currency(match):
            whole = match.group(1)
            decimal = match.group(2)
            unit = match.group(3)
            decimal_words = ' '.join([self.digits[int(d)] for d in decimal])
            unit_map = {'k': 'nghìn', 'm': 'triệu', 'b': 'tỷ'}
            unit_word = unit_map.get(unit.lower(), unit)
            return f"{whole} phẩy {decimal_words} {unit_word}"
        
        text = re.sub(r'(\d+)[.,](\d+)\s*([kmb])\b', decimal_currency, text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*k\b', r'\1 nghìn', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*m\b', r'\1 triệu', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*b\b', r'\1 tỷ', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*đ\b', r'\1 đồng', text)
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*vnd\b', r'\1 đồng', text, flags=re.IGNORECASE)
        text = re.sub(r'\$\s*(\d+(?:[.,]\d+)?)', r'\1 đô la', text)
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*\$', r'\1 đô la', text)
        return text
    
    def _normalize_percentage(self, text):
        """Convert percentage to words."""
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*%', r'\1 phần trăm', text)
        return text
    
    def _normalize_units(self, text):
        """Convert measurement units to words."""
        def expand_compound_with_number(match):
            number = match.group(1)
            unit1 = match.group(2).lower()
            unit2 = match.group(3).lower()
            full_unit1 = self.units.get(unit1, unit1)
            full_unit2 = self.units.get(unit2, unit2)
            return f"{number} {full_unit1} trên {full_unit2}"
        
        def expand_compound_without_number(match):
            unit1 = match.group(1).lower()
            unit2 = match.group(2).lower()
            full_unit1 = self.units.get(unit1, unit1)
            full_unit2 = self.units.get(unit2, unit2)
            return f"{full_unit1} trên {full_unit2}"
        
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*([a-zA-Zμµ²³°]+)/([a-zA-Zμµ²³°0-9]+)\b', 
                     expand_compound_with_number, text)
        text = re.sub(r'\b([a-zA-Zμµ²³°]+)/([a-zA-Zμµ²³°0-9]+)\b', 
                     expand_compound_without_number, text)
        
        sorted_units = sorted(self.units.items(), key=lambda x: len(x[0]), reverse=True)
        for unit, full_name in sorted_units:
            if unit == 'm':
                pattern = r'(\d+(?:[.,]\d+)?)\s*' + re.escape(unit) + r'((?:\d+)?)\b'
                
                def make_suffix(match):
                    r"""If match group 2, put it in the end"""
                    before_unit = match.group(1).lower()
                    after_unit = match.group(2).lower()
                
                    suffix = f" {after_unit}" if after_unit else ""
                    return f"{before_unit} {full_name}{suffix}"
                
                text = re.sub(pattern, make_suffix, text, flags=re.IGNORECASE)

            else:
                pattern = r'(\d+(?:[.,]\d+)?)\s*' + re.escape(unit) + r'\b'
                text = re.sub(pattern, rf'\1 {full_name}', text, flags=re.IGNORECASE)
        
        for unit, full_name in sorted_units:
            if any(c in unit for c in '²³°'):
                pattern = r'\b' + re.escape(unit) + r'\b'
                text = re.sub(pattern, full_name, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_time(self, text):
        """Convert time notation to words with validation."""
        
        def validate_and_convert_time(match):
            """Validate time components before converting."""
            groups = match.groups()
            
            # HH:MM:SS format
            if len(groups) == 3:
                hour, minute, second = groups
                hour_int, minute_int, second_int = int(hour), int(minute), int(second)
                
                if not (0 <= hour_int <= 23):
                    return match.group(0)
                if not (0 <= minute_int <= 59):
                    return match.group(0)
                if not (0 <= second_int <= 59):
                    return match.group(0)
                
                return f"{hour} giờ {minute} phút {second} giây"
            
            # HH:MM or HHhMM format
            elif len(groups) == 2:
                hour, minute = groups
                hour_int, minute_int = int(hour), int(minute)
                
                if not (0 <= hour_int <= 23):
                    return match.group(0)
                if not (0 <= minute_int <= 59):
                    return match.group(0)
                
                return f"{hour} giờ {minute} phút"
            
            # HHh format
            else:
                hour = groups[0]
                hour_int = int(hour)
                
                if not (0 <= hour_int <= 23):
                    return match.group(0)
                
                return f"{hour} giờ"
        
        text = re.sub(r'(\d{1,2}):(\d{2}):(\d{2})', validate_and_convert_time, text)
        text = re.sub(r'(\d{1,2}):(\d{2})', validate_and_convert_time, text)
        text = re.sub(r'(\d{1,2})h(\d{2})', validate_and_convert_time, text)
        text = re.sub(r'(\d{1,2})h\b', validate_and_convert_time, text)
        
        return text
    
    def _normalize_date(self, text):
        """Convert date notation to words with validation."""
        
        def is_valid_date(day, month, year):
            """Check if date components are valid."""
            day, month, year = int(day), int(month), int(year)
            
            if not (1 <= day <= 31):
                return False
            if not (1 <= month <= 12):
                return False

            return True
        
        def date_to_text(match):
            day, month, year = match.groups()
            if is_valid_date(day, month, year):
                return f"ngày {day} tháng {month} năm {year}"
            return match.group(0)
        
        def date_iso_to_text(match):
            year, month, day = match.groups()
            if is_valid_date(day, month, year):
                return f"ngày {day} tháng {month} năm {year}"
            return match.group(0)
        
        def date_short_year(match):
            day, month, year = match.groups()
            full_year = f"20{year}" if int(year) < 50 else f"19{year}"
            if is_valid_date(day, month, full_year):
                return f"ngày {day} tháng {month} năm {full_year}"
            return match.group(0)
        
        text = re.sub(r'\bngày\s+(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b', 
                    lambda m: date_to_text(m).replace('ngày ngày', 'ngày'), text)
        text = re.sub(r'\bngày\s+(\d{1,2})[/\-](\d{1,2})[/\-](\d{2})\b', 
                    lambda m: date_short_year(m).replace('ngày ngày', 'ngày'), text)
        text = re.sub(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', date_iso_to_text, text)
        text = re.sub(r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b', date_to_text, text)
        text = re.sub(r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2})\b', date_short_year, text)
        
        return text
    
    def _normalize_phone(self, text):
        """Convert phone numbers to digit-by-digit reading."""
        def phone_to_text(match):
            phone = match.group(0)
            phone = re.sub(r'[^\d]', '', phone)
            
            if phone.startswith('84') and len(phone) >= 10:
                phone = '0' + phone[2:]
            
            if 10 <= len(phone) <= 11:
                words = [self.digits[int(d)] for d in phone]
                return ' '.join(words) + ' '
            
            return match.group(0)
        
        text = re.sub(r'(\+84|84)[\s\-\.]?\d[\d\s\-\.]{7,}', phone_to_text, text)
        text = re.sub(r'\b0\d[\d\s\-\.]{8,}', phone_to_text, text)
        return text
    
    def _normalize_versions(self, text):
        """Convert version numbers like 1.0.4 to words."""
        def version_to_text(match):
            parts = match.group(0).split('.')
            # Convert each part to words if it's a number, or keep if not
            # But for versions, usually we just want the digits or numbers
            # The user requested "1 chấm 0 chấm 4"
            return ' chấm '.join(parts)
        
        # Match sequences of numbers separated by dots (at least 2 dots to be sure it's a version)
        # e.g., 1.0.4, 17.21.1, 192.168.1.1
        text = re.sub(r'\b\d+(?:\.\d+){1,}\b', version_to_text, text)
        return text
    
    def _normalize_numbers(self, text):
        text = re.sub(r'(\d+(?:[,.]\d+)?)%', lambda m: f'{m.group(1)} phần trăm', text)
        text = re.sub(r'(\d{1,3})(?:\.(\d{3}))+', lambda m: m.group(0).replace('.', ''), text)
    
        def decimal_to_words(match):
            whole = match.group(1)
            decimal = match.group(2)
            decimal_words = ' '.join([self.digits[int(d)] for d in decimal])
            separator = 'phẩy' if ',' in match.group(0) else 'chấm'
            return f"{whole} {separator} {decimal_words}"
        
        text = re.sub(r'(\d+),(\d+)', decimal_to_words, text)
        text = re.sub(r'(\d+)\.(\d{1,2})\b', decimal_to_words, text)
        
        return text
    
    def _read_two_digits(self, n):
        """Read two-digit numbers in Vietnamese."""
        if n < 10:
            return self.digits[n]
        elif n == 10:
            return "mười"
        elif n < 20:
            if n == 15:
                return "mười lăm"
            return f"mười {self.digits[n % 10]}"
        else:
            tens = n // 10
            ones = n % 10
            if ones == 0:
                return f"{self.digits[tens]} mươi"
            elif ones == 1:
                return f"{self.digits[tens]} mươi mốt"
            elif ones == 5:
                return f"{self.digits[tens]} mươi lăm"
            else:
                return f"{self.digits[tens]} mươi {self.digits[ones]}"
    
    def _read_three_digits(self, n):
        """Read three-digit numbers in Vietnamese."""
        if n < 100:
            return self._read_two_digits(n)
        
        hundreds = n // 100
        remainder = n % 100
        result = f"{self.digits[hundreds]} trăm"
        
        if remainder == 0:
            return result
        elif remainder < 10:
            result += f" lẻ {self.digits[remainder]}"
        else:
            result += f" {self._read_two_digits(remainder)}"
        
        return result
    
    def _convert_number_to_words(self, num):
        """Convert a number to Vietnamese words."""
        if num == 0:
            return "không"
        
        if num < 0:
            return f"âm {self._convert_number_to_words(-num)}"
        
        if num >= 1000000000:
            billion = num // 1000000000
            remainder = num % 1000000000
            result = f"{self._read_three_digits(billion)} tỷ"
            if remainder > 0:
                result += f" {self._convert_number_to_words(remainder)}"
            return result
        
        elif num >= 1000000:
            million = num // 1000000
            remainder = num % 1000000
            result = f"{self._read_three_digits(million)} triệu"
            if remainder > 0:
                result += f" {self._convert_number_to_words(remainder)}"
            return result
        
        elif num >= 1000:
            thousand = num // 1000
            remainder = num % 1000
            result = f"{self._read_three_digits(thousand)} nghìn"
            if remainder > 0:
                if remainder < 10:
                    result += f" không trăm lẻ {self.digits[remainder]}"
                elif remainder < 100:
                    result += f" không trăm {self._read_two_digits(remainder)}"
                else:
                    result += f" {self._read_three_digits(remainder)}"
            return result
        
        else:
            return self._read_three_digits(num)
    
    def _number_to_words(self, text):
        """Convert all remaining numbers to words."""
        def convert_number(match):
            num = int(match.group(0))
            return self._convert_number_to_words(num)
        
        text = re.sub(r'\b\d+\b', convert_number, text)
        return text
    
    def _normalize_special_chars(self, text):
        """Handle special characters."""
        # Remove quotes first to avoid creating spaces before commas
        text = text.replace('"', '')
        text = text.replace("'", '')
        text = text.replace(''', '')
        text = text.replace(''', '')
        text = text.replace('"', '')
        text = text.replace('"', '')
        
        text = text.replace('&', ' và ')
        text = text.replace('+', ' cộng ')
        text = text.replace('=', ' bằng ')
        text = text.replace('#', ' thăng ')
        # Handle parentheses/brackets as natural pauses: (text) -> , text ,
        text = re.sub(r'[\(\[\{]\s*(.*?)\s*[\)\]\}]', r', \1, ', text)
        
        # Remaining individual brackets or parens
        text = re.sub(r'[\[\]\(\)\{\}]', ' ', text)
        
        # Paired dashes (like parentheses): - text - -> , text ,
        text = re.sub(r'(?:\s+|^)[-–—]\s*(.*?)\s*[-–—](?:\s+|$)', r', \1 , ', text)
        
        # Single dashes used as punctuation (with spaces) -> comma
        text = re.sub(r'\s+[-–—]+\s+', ', ', text)
        
        # Dashes at the start of a line (bullet points) -> comma
        text = re.sub(r'^[-–—]+\s+', ', ', text)
        
        # Collapse multiple commas and surrounding spaces (remove spaces before AND after commas)
        text = re.sub(r'\s*,\s*', ', ', text)
        text = re.sub(r',\s*,+', ',', text)  # Remove duplicate commas
        
        text = re.sub(r'\.{2,}', ' ', text)
        text = re.sub(r'\s+\.\s+', ' ', text)
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ.,!?;:@%_]', ' ', text)
        return text
    
    def _normalize_whitespace(self, text):
        """Normalize whitespace."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text


if __name__ == "__main__":
    normalizer = VietnameseTTSNormalizer()
    
    test_texts = [
        "Chỉ cần thay đổi một dấu thanh, ý nghĩa của từ đã hoàn toàn khác biệt. Ví dụ như \"ma\", \"má\", \"mà\", \"mả\", \"mã\", \"mạ\" – đây chính là \"bài toán khó\" mà các kỹ sư công nghệ phải giải quyết để tạo ra một giọng đọc tự nhiên như người bản xứ.",
        "Phiên bản hiện tại là 1.0.4 và địa chỉ IP của tôi là 192.168.1.1"
    ]
    
    print("=" * 80)
    print("VIETNAMESE TTS NORMALIZATION TEST (WITH EN TAG)")
    print("=" * 80)
    
    for text in test_texts:
        print(f"\n📝 Input: {text}")
        normalized = normalizer.normalize(text)
        print(f"🎵 Output: {normalized}")
        print("-" * 80)