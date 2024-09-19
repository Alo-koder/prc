'''
A module for plotting the analysis results.
'''


import matplotlib as mpl
from matplotlib import pyplot as plt
from config import props, filenames
import os


def save_without_overwrite(fig: mpl.figure.Figure, filename:str):
    if os.path.exists(filenames.notes+filename):
        return
    fig.savefig(filenames.notes+filename)


def current_and_emsi(data, data_raw):
    fig, ax = plt.subplots()
    ax.set_title('Current vs time')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Current Density [mA/cm$^2$]')
    ax.plot(data.t, data.emsi_corrected+1.0, label='emsi')
    ax.plot(data_raw.t, data_raw.I, label='current')
    ax.plot(data.t, data.I, label='treated')
    # plt.scatter(pert_times, np.interp(pert_times, data.t, data.I)) # show perturbations
    fig.legend(loc=1)
    fig.show()
    save_without_overwrite(fig, 'current_and_emsi_waveform.png')


def period_vs_time(cycles, perts):
    fig, axs = plt.subplots(2, sharex=True)

    axs[0].plot(cycles.start, cycles.duration)
    axs[0].plot(perts.time, perts.expected_period)
    axs[0].set_ylabel('Period [s]')

    axs[1].scatter(perts.time, perts.phase, marker = 'x')
    axs[1].set_ylabel(r'$\phi$ at Perturbation')

    fig.suptitle('Period Drift and Perturbation Distribution')
    fig.supxlabel('Time [s]')
    
    fig.show()
    save_without_overwrite(fig, 'period_vs_time.png')


def amplitude_vs_time(cycles, perts):
    fig, axs = plt.subplots(2, sharex=True)

    axs[0].plot(cycles.start, cycles.amplitude)
    axs[0].plot(cycles.start, cycles.expected_amplitude)
    axs[0].set_ylabel('Amplitude')

    axs[1].scatter(perts.time, perts.phase, marker = 'x')
    axs[1].set_ylabel(r'$\phi$ at Perturbation')

    fig.suptitle('Amplitude Drift and Perturbation Distribution')
    fig.supxlabel('Time [s]')

    fig.show()
    save_without_overwrite(fig, 'amplitude_vs_time.png')


def prc_current(perts, exp_summary):
    fig, ax = plt.subplots()
    fig.suptitle(exp_summary)
    ax.axhline(0, ls='--', c='grey')
    ax.scatter(perts.phase, perts.response, c=perts.time)
    ax.set_xlabel("phase wrt. current max.")
    ax.set_ylabel(r"$\frac{T_1+T_2}{2T_0}$")

    fig.show()
    save_without_overwrite(fig, 'prc_current.png')


def prc_emsi(perts, exp_summary):
    fig, ax = plt.subplots()
    fig.suptitle(exp_summary)
    ax.axhline(0, ls='--', c='grey')
    ax.scatter(perts.corrected_phase, perts.response, c=perts.time)
    ax.set_xlabel("phase wrt. emsi min.")
    ax.set_ylabel(r"$\frac{T_1+T_2}{2T_0}$")

    fig.show()
    save_without_overwrite(fig, 'prc_emsi.png')


def amplitude_response(perts, exp_summary):
    fig, ax = plt.subplots()
    ax.set_title(' '.join(['Amplitude response', *exp_summary.split(' ')[1:]]))
    ax.scatter(perts.phase, perts.amp_response_1, c=perts.time, picker=True)
    ax.set_xlabel("phase wrt. current max.")
    ax.set_ylabel('Amplitude response (fractional)')
    ax.axhline(0, ls='--', c='grey')
    save_without_overwrite(fig, 'amp_response.png')


def prc(perts, example_period, exp_summary):
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle(exp_summary)
    fig.supxlabel('phase wrt. current max.')
    axs[0].axhline(0)
    axs[0].scatter(perts.phase, perts.response, c=perts.time)
    axs[0].set_ylabel('Phase Response')
    axs[1].plot(example_period.phase, example_period.I_real)
    axs[1].set_ylabel(r'J [mA/cm$^2$]')
    axs[1].plot(example_period.phase, example_period.emsi_corrected+0.25)
    save_without_overwrite(fig, 'prc.png')


def interactive_prc(perts, data):
    fig, ax = plt.subplots()
    fig.suptitle('Interactive PRC')
    ax.axhline(0, ls='--', c='grey')
    ax.scatter(perts.phase, perts.response, c=perts.time, picker=True)
    fig_phase, ax_phase = plt.subplots()
    fig_current, ax_current = plt.subplots()

    def on_pick(event) -> tuple[mpl.figure.Figure, mpl.figure.Axes]:
        ax_phase.cla()
        ax_current.cla()

        pert_time = perts.time.iloc[event.ind[0]]
        print(pert_time)
        data_before = data[(data.t > pert_time - props.max_period) & (data.t < pert_time)]
        data_after = data[(data.t > pert_time) & (data.t < pert_time + 2*props.max_period)]

        ax_phase.plot(data_before.I_real, data_before.emsi)
        ax_phase.plot(data_after.I_real, data_after.emsi)
        ax_phase.scatter(data_before.I_real.iloc[-1], data_before.emsi.iloc[-1], c='r')
        fig_phase.suptitle(rf'Perturbation at $\phi$ = {perts.phase.iloc[event.ind[0]]:.2f} -- phase space')
        fig_phase.supxlabel(r'current [mA/cm$^2$]')
        fig_phase.supylabel(r'emsi signal [$\xi$]')
        fig_phase.canvas.draw()

        ax_current.plot(data_before.t, data_before.I_real)
        ax_current.plot(data_after.t, data_after.I_real)
        ax_current.scatter(data_before.t.iloc[-1], data_before.I_real.iloc[-1], c='r')
        fig_current.suptitle(rf'Perturbation at $\phi$ = {perts.phase.iloc[event.ind[0]]:.2f} -- current')
        fig_current.supxlabel('time [s]')
        fig_current.supylabel(r'current [mA/cm$^2$]')
        fig_current.canvas.draw()

    id = fig.canvas.mpl_connect('pick_event', on_pick)
    fig.show()
    fig.canvas.mpl_disconnect(id)


def prc_full(perts, exp_summary, things_to_plot=['response_1', 'response_2', 'response_3', 'response_4']):
    fig, axs = plt.subplots(2,2, sharex = True, sharey=True, figsize = (14, 9))
    for (thing, ax) in zip(things_to_plot, axs.flatten()):
        ax.scatter(perts.phase, perts[thing], c=perts.time)
        ax.set_title(thing)
        ax.axhline(0, ls='--', c='grey')
    fig.suptitle('Full '+exp_summary)
    fig.supxlabel(r'$\phi$ wrt. current max.')
    fig.supylabel(r'$\Delta\phi$')
    fig.tight_layout()
    fig.show()
    save_without_overwrite(fig, 'prc_full.png')

def amp_response_full(perts, exp_summary, things_to_plot=['basis_amplitude', 'amp_response_1', 'amp_response_2', 'amp_response_3']):
    fig, axs = plt.subplots(2,2, sharex = True, sharey=True, figsize = (14, 9))
    for (thing, ax) in zip(things_to_plot, axs.flatten()):
        ax.scatter(perts.phase, perts[thing], c=perts.time)
        ax.set_title(thing)
        ax.axhline(0, ls='--')
        ax.set_xlim(-0.03,1.03)
    fig.suptitle(exp_summary)
    fig.supxlabel(r'$\phi$ wrt. current max.')
    fig.supylabel(r'$\Delta\phi$')
    fig.tight_layout()
    fig.show()
    save_without_overwrite(fig, 'amp_response_full.png')