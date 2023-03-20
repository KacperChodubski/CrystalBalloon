import math
import datetime
import data.ecmwf_data_collector

def find_bd(balloon_mass: str):
        burst_diameters = {
            'h100': 180,
            'h300': 380,
            'h350': 410,
            'h500': 500,
            'h600': 580,
            'h800': 680,
            'h1000': 750,
            'h1200': 850,
            'h1600': 1050,
            'h2000': 1100,
            'h3000': 12.50,
        }
        bd = burst_diameters[balloon_mass] / 100
        return bd
    
def find_cd(balloon_mass: str):
    cds = {
        "h200": 0.25,
        "h100": 0.25,
        "h300": 0.25,
        "h350": 0.25,
        "h500": 0.25,
        "h600": 0.30,
        "h750": 0.30,
        "h800": 0.30,
        "h950": 0.30,
        "h1000": 0.30,
        "h1200": 0.25,
        "h1500": 0.25,
        "h1600": 0.25,
        "h2000": 0.25,
        "h3000": 0.25,
    }
    cd = cds[balloon_mass]
    return cd

def calculate_burst_altitude(balloon_mass: str, payload_mass: float, ascent_rate: float):

        rho_g = 0.1786 # for hel

        
        mb = float(balloon_mass[1:]) / 1000
        mp = payload_mass / 1000
        

        bd = find_bd(balloon_mass) 
        cd = find_cd(balloon_mass) 
        rho_a = 1.2050                  # air density
        adm = 7238.3                    # air density model
        ga = 9.80665                    # gravity aceleration
        tar = ascent_rate               # target ascenting rate


        burst_volume = 4/3 * math.pi * (bd/2)**3

        a = ga * (rho_a - rho_g) * (4.0 / 3.0) * math.pi
        b = -0.5 * math.pow(tar, 2) * cd * rho_a * math.pi
        c = 0
        d = - (mp + mb) * ga
        f = (((3*c)/a) - (math.pow(b, 2) / math.pow(a,2)) / 3.0)
        g = (((2*math.pow(b,3))/math.pow(a,3)) - ((9*b*c)/(math.pow(a,2))) + ((27*d)/a)) / 27.0
        h = (math.pow(g,2) / 4.0) + (math.pow(f,3) / 27.0)
        R = (-0.5 * g) + math.sqrt(h)
        S = math.pow(R, 1.0/3.0)
        T = (-0.5 * g) - math.sqrt(h)
        U = math.pow(T, 1.0/3.0)

        launch_radius = (S+U) - (b/(3*a))

        launch_volume = (4.0/3.0) * math.pi * math.pow(launch_radius, 3)

        volume_ratio = launch_volume / burst_volume

        burst_altitude = -(adm) * math.log(volume_ratio)

        print(f'Predicted burst altitude: {burst_altitude}')
        print(f'Required launch volume: {launch_volume}')

        return burst_altitude

def calculate_pressure(ecmwf: data.ecmwf_data_collector.ECMWF_data_collector, lat, lon, alt, alt0, time_for_pred: datetime.date):
    Tb, _, _ = ecmwf.get_data(lat, lon, 1000, time_for_pred)
    msl = ecmwf.get_msl(lat, lon, time_for_pred)
    Lb = -0.0065
    g0 = 9.80665
    R = 8.31432
    M = 0.0289644
    pressure: float
    if alt < 10000:
        pressure = msl * (1 + (Lb/Tb)*(alt - alt0))**((-g0 * M)/(R*Lb))
    elif alt > 10000:
        pressure = msl * math.exp((-g0 * M * (alt - alt0)) / (R*Tb))

    return pressure