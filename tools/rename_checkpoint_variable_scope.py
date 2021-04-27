import argparse
import tensorflow as tf


parser = argparse.ArgumentParser()

# Input paths
parser.add_argument('--checkpoint_load_path',
    type=str, required=True, help='Path to load checkpoint')
parser.add_argument('--checkpoint_save_path',
    type=str, required=True, help='Path to save checkpoint with renamed variable scope')
parser.add_argument('--variable_scope',
    type=str, required=True, help='Existing variable scope to rename')
parser.add_argument('--new_variable_scope',
    type=str, required=True, help='New variable scope')

args = parser.parse_args()


with tf.Session() as session:
    # Load checkpoint and iterate through variables
    for variable_name, _ in tf.contrib.framework.list_variables(args.checkpoint_load_path):

        variable = tf.contrib.framework.load_variable(
            args.checkpoint_load_path,
            variable_name)

        variable_renamed = variable_name

        # Rename variable scope
        if args.variable_scope in variable_name:

            variable_renamed = variable_renamed.replace(args.variable_scope, args.new_variable_scope)

            print('Renaming {} to {}'.format(variable_name, variable_renamed))

        variable = tf.Variable(variable, name=variable_renamed)

    # Save the variables
    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())
    saver.save(session, args.checkpoint_save_path)
