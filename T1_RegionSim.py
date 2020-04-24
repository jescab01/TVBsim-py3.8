from tvb.simulator.lab import *
import numpy
from matplotlib.pyplot import *

oscilator = models.Generic2dOscillator()

white_matter = connectivity.Connectivity.from_file("connectivity_76.zip")
white_matter.speed = numpy.array([4.0])

white_matter_coupling = coupling.Linear(a=numpy.array([0.0154]))

heunint = integrators.HeunDeterministic(dt=2 ** -6)

mon_raw = monitors.Raw() # I calculated its period as T=secs/points=1024/65536=0.0156 - Default period?
mon_tavg = monitors.TemporalAverage(period=2 ** -2)

what_to_watch = (mon_raw, mon_tavg)

## Simulator
sim = simulator.Simulator(model=oscilator, connectivity=white_matter,
                          coupling=white_matter_coupling,
                          integrator=heunint, monitors=what_to_watch)

sim.configure()  ##Calculate information necessary for the simulation that draws on specific combinations of the components

## Run simulation
raw_data = []
raw_time = []
tavg_data = []
tavg_time = []

for raw, tavg in sim(simulation_length=2 ** 10):
    if not raw is None:
        raw_time.append(raw[0])
        raw_data.append(raw[1])

    if not tavg is None:
        tavg_time.append(tavg[0])
        tavg_data.append(tavg[1])

RAW = numpy.array(raw_data)
TAVG = numpy.array(tavg_data)

#Plot raw time series
figure(1)
plot(raw_time, RAW[:, 0, :, 0])
title("Raw -- State variable 0")

#Plot temporally averaged time series
figure(2)
plot(tavg_time, TAVG[:, 0, :, 0])
title("Temporal average")
