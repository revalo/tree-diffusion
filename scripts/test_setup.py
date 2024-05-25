from absl import app
from absl import flags

from td.environments import environments
from td.samplers import ConstrainedRandomSampler
from td.samplers.mutator import random_mutation, find_path

flags.DEFINE_string("environment", "csg2da", "Environment to evaluate.")
FLAGS = flags.FLAGS


def main(argv):
    env = environments[FLAGS.environment]()
    sampler = ConstrainedRandomSampler(env.grammar)
    expression = sampler.sample(env.grammar.start_symbol, 4, 4)
    print("Random expression:", expression)
    
    print("Mutating!")

    current_expression = expression
    for i in range(4):
        m = random_mutation(current_expression, env.grammar, sampler)
        print(m.pretty(current_expression))
        current_expression = m.apply(current_expression)

    print(current_expression)



if __name__ == "__main__":
    app.run(main)
