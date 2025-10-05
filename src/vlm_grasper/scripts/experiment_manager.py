#!/usr/bin/env python

import rospy
import os
import yaml
import roslib

class ExperimentManager:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('experiment_manager', anonymous=True)

        rospy.loginfo("Experiment Manager node started.")

        self.first_run = True


        
        # Get list of config files
        self.config_files = rospy.get_param('scenes', [])

        package_path = roslib.packages.get_pkg_dir('vlm_grasper')

        self.config_paths = [os.path.join(package_path, 'experiments', scene, 'config.yaml') for scene in self.config_files]
        self.current_experiment = 0
        rospy.set_param("experiment_number", self.current_experiment)

        # Initialize the 'finished' parameter to False
        rospy.set_param('finished', True)
        
        # Start the main loop
        self.run()


    def load_config(self, config_file):
        # Load the parameters from the YAML file directly into ROS params
        rospy.loginfo(f"Loading config file: {config_file}")

        try:
            with open(config_file, 'r') as file:
                config_params = yaml.safe_load(file)
                for param, value in config_params.items():
                    rospy.set_param(param, value)
                    rospy.loginfo(f"Set parameter: {param} = {value}")
            rospy.loginfo("Config loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Failed to load config file: {e}")

    def save_params_to_file(self):
        params = [
            "n_grasps_beginning",
            "n_grasps_geometry_mask",
            "n_grasps_surface_normal",
            "n_grasps_good_score",
            "n_grasps_reduced_to_100",
            "n_grasps_feasible",
            "n_grasps_chosen",
            "t_start",
            "t_pc_processed",
            "t_edge_grasp_completed",
            "t_filtering_completed",
            "t_rendering_completed",
            "t_end"
        ]

        path = roslib.packages.get_pkg_dir('vlm_grasper') + "/experiments/" + rospy.get_param('scene', 'unknown') + "/results"

        if not os.path.exists(path):
            os.makedirs(path)
        
        results_file = os.path.join(path, "results.yaml")

        with open(results_file, 'w') as file:
            for param in params:
                value = rospy.get_param(param, None)
                if value is not None:
                    file.write(f"{param}: {value}\n")

    def run(self):
        while not rospy.is_shutdown():
            # Check the 'finished' parameter
            finished = rospy.get_param("finished", True)
            

            if finished:

                self.save_params_to_file()
                if not self.first_run:
                    rospy.loginfo("Sleeping for 60 seconds...")
                    rospy.sleep(60)
                    rospy.loginfo("Woke up after sleeping.")
                    rospy.loginfo("Experiment finished, loading next config...")
                    self.first_run = False
                

                # Load the next config file if available
                if self.current_experiment < len(self.config_paths):
                    config_file = self.config_paths[self.current_experiment]
                    self.load_config(config_file)

                    scene = rospy.get_param('scene', '')
                    
                    rospy.loginfo(f"Starting experiment for scene: {scene}")
                    
                    # Reset the 'finished' parameter
                    rospy.set_param('finished', False)

                    rospy.set_param("point_cloud_transformed", False)
                    rospy.set_param("edge_grasp_finished", False)
                    rospy.set_param("grasp_image_renderer_finished", False)

                    
                    # Move to the next experiment
                    self.current_experiment += 1
                    rospy.set_param("experiment_number", self.current_experiment)

                else:
                    rospy.loginfo("All experiments are completed.")
                    rospy.signal_shutdown("Experiment sequence finished")
                    
            # Sleep to prevent busy-waiting
            rospy.sleep(1)

if __name__ == '__main__':
    try:
        ExperimentManager()
    except rospy.ROSInterruptException:
        #Print an error message if the node is interrupted
        rospy.logerr("Experiment Manager node was interrupted.")
