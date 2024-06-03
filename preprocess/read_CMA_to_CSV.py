class Typhoon:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.records = []
        self.reached_cat_2 = False

    def add_record(self, record):
        self.records.append(record)
        if record['category'] >= 2 and record['category'] != 9:
            self.reached_cat_2 = True


def parse_path_record(line):
    parts = line.split()
    date = parts[0]
    hour = int(date[-2:])  # 提取小时部分
    if hour not in [0, 6, 12, 18]:
        return None  # 如果不是目标时间，返回 None
    wind_speed = int(parts[5])
    category = int(parts[1])
    if category == 9:
        return None
    lat = float(parts[2]) / 10
    lon = float(parts[3]) / 10
    pressure = int(parts[4])
    return {'date': date, 'category': category, 'lat': lat, 'lon': lon, 'pressure': pressure, 'wind_speed': wind_speed}


def parse_file(filename):
    typhoons = []
    current_typhoon = None
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('66666'):
                parts = line.split()
                id = parts[4]
                if id != '0000':  # Skip typhoon with ID '0000'
                    name = parts[7].strip()
                    current_typhoon = Typhoon(id, name)
                    typhoons.append(current_typhoon)  # 正确的位置：将新的台风添加到列表
                else:
                    current_typhoon = None
            else:
                if current_typhoon is not None:
                    record = parse_path_record(line)
                    if record is not None:  # 检查 record 是否有效
                        # Check if record is within the specified geographic boundaries
                        if 0 <= record['lat'] <= 70 and 100 <= record['lon'] <= 180:
                            current_typhoon.add_record(record)
        # Only keep typhoons that have reached category 2 strength and have at least one record
        typhoons = [typhoon for typhoon in typhoons if typhoon.reached_cat_2 and len(typhoon.records) > 0]

    return typhoons


def save_to_csv(typhoons, filename):
    import csv
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Assuming you want to write headers
        writer.writerow(['ID', 'Name', 'Date', 'Category', 'Latitude', 'Longitude', 'Pressure', 'Wind Speed'])
        for typhoon in typhoons:
            for record in typhoon.records:
                writer.writerow(
                    [typhoon.id, typhoon.name, record['date'], record['category'], record['lat'], record['lon'],
                     record['pressure'], record['wind_speed']])


def main():
    files = []
    for i in range(2020, 2024):
        files.append(f'./data_file/CMA_typhoon/CH{str(i)}BST.txt')
    all_typhoons = []
    for file in files:
        all_typhoons.extend(parse_file(file))
    save_to_csv(all_typhoons, './data_file/typhoons.csv')

    # Here, you can process and save the `all_typhoons` list as needed
    # For example, print out or save to a structured file like CSV or JSON


if __name__ == "__main__":
    main()
