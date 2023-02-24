from networks.Reference_Trajectory_Generation import ReferenceTrajectoryData, ReferenceTrajectoryUpdate, \
    TrajectoryMappingNetwork

list_of_subjects = ['DS01', 'DS02', 'DS04', 'DS05', 'DS06', 'DS07']
dominant_leg = {'DS01': 'R', 'DS02': 'L', 'DS04': 'L', 'DS05': 'R', 'DS06': 'R', 'DS07': 'L'}
list_of_speeds = ['09', '10', '11', '12', '13', '14', '15', '16', '17', '18', 'R']
list_of_muscles = ["ESL-L", "ESL-R", "ESI-L", "ESI-R", "MF-L", "MF-R", "RF", "VM", "VL", "BF", "ST", "TA", "SO", "GM",
                   "GL"]
list_joint_angles = ["HipAngles", "KneeAngles", "AnkleAngles"]

n_classes = [5, 10]
window_step = 5
window_lengths = [500, 1000]

for window_length in window_lengths:
    for nclass in n_classes:
        trajectory_data = ReferenceTrajectoryData(['DS01'], list_of_speeds[0:-1], list_of_muscles, 'Knee',
                                                  window_length=window_length, label_type='percentage based classes',
                                                  number_of_classes=nclass, window_step=window_step, batch_size=64)
        network = TrajectoryMappingNetwork(learning_rate=0.00001, epochs=500,
                                           saved_model_name='percentage_based_classes_' + str(window_length) + '_' +
                                                            str(nclass), model_type='classifier',
                                           window_length=window_length, number_of_classes=nclass)
        network.train_network(trajectory_data.x_train, trajectory_data.y_train, trajectory_data.x_val,
                              trajectory_data.y_val)
        network.test_network(trajectory_data.x_test, trajectory_data.y_test)

