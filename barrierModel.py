import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from functs import *
from plottingFuncts import *
import os

def main():

    with open('config.json', 'r') as f:
        config = json.load(f)

    trajectory_settings = config['trajectory_settings']
    output_settings = config['output_settings']
    debug_settings = config.get('debug_settings', {})

    # create the output directory if it doesn't exist

    output_dir = output_settings.get('output_directory', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    

    data = simulate_line_kmc(
        N=trajectory_settings.get('N_node', 50),
        b=trajectory_settings.get('peierls_distance', 1.0),
        dx=trajectory_settings.get('dx', 1.0),
        T=trajectory_settings.get('temperatureT', 300),
        stress=trajectory_settings.get('stress', 0.1),
        mu_n=trajectory_settings.get('mu_n', 0.2),
        sig_n=trajectory_settings.get('sig_n', 0.1),
        mu_p=trajectory_settings.get('mu_p', 0.05),
        sig_p=trajectory_settings.get('sig_p', 0.05),
        Gamma_n=trajectory_settings.get('Gamma_n', 0.5),
        Gamma_p=trajectory_settings.get('Gamma_p', 0.1),
        Omega_n=trajectory_settings.get('Omega_n', 1.0),
        Omega_p=trajectory_settings.get('Omega_p', 1.0),
        steps=trajectory_settings.get('n_steps', 300)
    )


    x = np.arange(trajectory_settings.get('N_node', 50)+1) * trajectory_settings.get('dx', 1.0)
    y_snapshots = data['y_snapshots']


    fig, ax = plot_line_trajectory_colored(x, y_snapshots)

    if output_settings.get("save_plots", False):
        fig.savefig(
            f"{output_dir}/{output_settings.get('plot_filename', 'line_trajectory.png')}",
            dpi=300,
            bbox_inches="tight"
        )

    if output_settings.get("show_plots", True):
        plt.show()
    else:
        plt.close(fig)


    if output_settings.get('save_trajectory', False):
        # convert any numpy arrays to native Python lists so JSON can serialize them
        if isinstance(data.get('y_snapshots'), np.ndarray):
            y_snapshots_list = data['y_snapshots'].tolist()
        else:
            y_snapshots_list = [np.asarray(y).tolist() for y in data.get('y_snapshots', [])]

        event_times_list = np.asarray(data.get('event_times', [])).tolist()

        output_data = {
            'y_snapshots': y_snapshots_list,
            'event_times': event_times_list
        }
        # save to the output directory
        traj_path = os.path.join(output_dir, output_settings.get('trajectory_filename', 'simulation_data.json'))
        with open(traj_path, 'w') as f: # save to the output directory
            json.dump(output_data, f)

    if output_settings.get('save_animation', False):
        animation_filename = output_settings.get('animation_filename', 'line_trajectory.gif')
        animation_path = os.path.join(output_dir, animation_filename)
        create_animation(x, y_snapshots, animation_path)
    

    if output_settings.get('return_velocity_data', False):
        velocity = return_velocity(data.get('event_times', []), trajectory_settings.get('dx', 1.0))
        velocity_data = {'velocity': velocity}
        velocity_path = os.path.join(output_dir, output_settings.get('velocity_filename', 'velocity_data.json'))
        with open(velocity_path, 'w') as f:
            json.dump(velocity_data, f)

    if output_settings.get('return_kink_data', False):
        num_kinks = return_num_kinks(data.get('y_snapshots', []))
        if isinstance(num_kinks, np.ndarray):
            num_kinks = num_kinks.tolist()
        kink_data = {'num_kinks': num_kinks}
        kink_path = os.path.join(output_dir, output_settings.get('kink_filename', 'kink_data.json'))
        with open(kink_path, 'w') as f:
            json.dump(kink_data, f)
            
    if output_settings.get('plot_velocity_data', False):
        fig, ax = plot_velocity(data.get('event_times', []), trajectory_settings.get('dx', 1.0))
        plt.show()
        
    if output_settings.get('plot_kink_data', False):
        fig, ax = plot_num_kinks(data.get('event_times', []),data.get('y_snapshots', []))
        plt.show()
        
    if debug_settings.get('output_curvature_animation', False):
        print('Generating curvature annotation animation...')
        curvature_animation_filename = debug_settings.get('curvature_animation_filename', 'curvature_animation.gif')
        curvature_animation_path = os.path.join(output_dir, curvature_animation_filename)
        visualize_animation_with_curvature_annotations(
            x, data.get('y_snapshots', []),
            filename=curvature_animation_path,
            interval=100
        )
        
    if debug_settings.get('output_deviation_animation', False):
        print('Generating deviation annotation animation...')
        deviation_animation_filename = debug_settings.get('deviation_animation_filename', 'deviation_animation.gif')
        deviation_animation_path = os.path.join(output_dir, deviation_animation_filename)
        visualize_animation_with_deviation_annotations(
            x, data.get('y_snapshots', []),
            filename=deviation_animation_path,
            interval=100
        )
        
    if debug_settings.get('output_rateCalculation_animation', False):
        print('Generating rate calculation annotation animation...')
        rateCalculation_animation_filename = debug_settings.get('rateCalculation_animation_filename', 'rateCalculation_animation.gif')
        rateCalculation_animation_path = os.path.join(output_dir, rateCalculation_animation_filename)
        Gn = np.random.normal(
            trajectory_settings.get('mu_n', 0.2),
            trajectory_settings.get('sig_n', 0.1),
            trajectory_settings.get('N_node', 50) + 1
        )
        Gp = np.random.normal(
            trajectory_settings.get('mu_p', 0.05),
            trajectory_settings.get('sig_p', 0.05),
            trajectory_settings.get('N_node', 50)
        )
        T = trajectory_settings.get('temperatureT', 300)
        stress = trajectory_settings.get('stress', 0.1)
        b = trajectory_settings.get('peierls_distance', 1.0)
        Gamma_n = trajectory_settings.get('Gamma_n', 0.5)
        Gamma_p = trajectory_settings.get('Gamma_p', 0.1)
        dx = trajectory_settings.get('dx', 1.0)
        Omega_n = trajectory_settings.get('Omega_n', 1.0)
        Omega_p = trajectory_settings.get('Omega_p', 1.0)
        y_snapshots = data.get('y_snapshots', [])
        visualize_animation_with_rateCalculation_annotations(
            x, y_snapshots, Gn, Gp, T, stress, b, Gamma_n, Gamma_p, dx, Omega_n, Omega_p, filename=rateCalculation_animation_path, interval=100
        )
        
                
                
    return()

if __name__ == "__main__":
    main()