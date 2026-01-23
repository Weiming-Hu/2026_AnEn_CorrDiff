from datetime import datetime, timedelta
import subprocess

start_date = datetime(2023, 3, 11)
end_date = datetime(2024, 3, 11)
lead_time = 168
time_str = "1200"
model = "graphcast"



while start_date <= end_date:
    date_str = start_date.strftime("%Y%m%d")
    output_filename = f"graphcast_{date_str}_{time_str}_{lead_time}h_cpu.grib"
    output_path = "E:\\graphcast\\" + output_filename

    command = [
        "ai-models",
        "--assets", "./assets/Graphcast",
        "--path", output_path,
        "--input", "cds",
        "--date", date_str,
        "--time", time_str,
        "--lead-time", str(lead_time),
        str(model)
    ]

    print(f"Running forecast for {date_str} -> {output_filename}")
    subprocess.run(command)
    start_date += timedelta(days=1)
