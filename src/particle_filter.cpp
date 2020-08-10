/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using std::string;
using std::vector;

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  // TODO: Set the number of particles
  num_particles = 100;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  particles.resize(num_particles);
  weights.resize(num_particles, 1.0);

  for(int i = 0; i < num_particles; ++i)
  {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  for(int i = 0; i < num_particles; ++i)
  {
    double xp, yp, thetap;
    
    if(fabs(yaw_rate) < 0.0001)
    {
      xp = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      yp = particles[i].y + velocity * delta_t * sin(particles[i].theta);
      thetap = particles[i].theta + delta_t * yaw_rate;
    }
    else
    {
      xp = particles[i].x + velocity * (
      	sin(particles[i].theta + delta_t * yaw_rate) - 
     	sin(particles[i].theta)) / yaw_rate;
      yp = particles[i].y + velocity * (
        cos(particles[i].theta) - 
        cos(particles[i].theta + delta_t * yaw_rate)) / yaw_rate;
      thetap = particles[i].theta + delta_t * yaw_rate;
    }
    
    particles[i].x = xp + dist_x(gen);
    particles[i].y = yp + dist_y(gen);
    particles[i].theta = thetap + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for(auto& ob : observations)
  {
    LandmarkObs temp;
    double mindist = std::numeric_limits<double>::max();

    for(const auto& pred : predicted)
    {
      double d = dist(pred.x, pred.y, ob.x, ob.y);
      if(d < mindist)
      {
        temp = pred;
        mindist = d;
      }
    }
    
    ob = temp;
  }
}

double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) 
{
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using zation terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for(int i = 0; i < num_particles; ++i)
  {
    double px = particles[i].x;
    double py = particles[i].y;
    double costheta = cos(particles[i].theta);
    double sintheta = sin(particles[i].theta);

    // compute predicted landmarks within sensor range
    vector<LandmarkObs> predicted;
    for(const auto& lm : map_landmarks.landmark_list)
    {
      LandmarkObs templm;
      templm.id = lm.id_i;
      templm.x = lm.x_f;
      templm.y = lm.y_f;
      double d = dist(templm.x, templm.y, px, py);
      if(d < sensor_range)
        predicted.push_back(templm);
    }

    std::vector<LandmarkObs> observations_p;
    // convert observation from sensor coordinate to map coordinate for every
    // partical
    for(const auto& obs : observations)
    {
      LandmarkObs tempob;
      tempob.id = obs.id;
      tempob.x = px + costheta * obs.x - sintheta * obs.y;
      tempob.y = py + sintheta * obs.x + costheta * obs.y;
      observations_p.push_back(tempob);
    }

    std::vector<LandmarkObs> observations_closest = observations_p;
    dataAssociation(predicted, observations_closest);

    // compute weight using observations_p and observations_closest;
    double weight = 1.0;
    for(auto j = 0; j < observations_closest.size(); ++j)
    {
      double w = multiv_prob(
        std_landmark[0],
        std_landmark[1],
        observations_closest[j].x, 
        observations_closest[j].y,
        observations_p[j].x,
        observations_p[j].y);

      weight *= w;
    }

    weights[i] = weight;
  }

  // normalize weight
  double wsum = 0.0;
  for(double w : weights)
    wsum += w;
  
  for(int i = 0; i < num_particles; ++i)
  {
    weights[i] = weights[i] / wsum;
    particles[i].weight = weights[i];
  }
  
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  std::discrete_distribution<int> distribution{weights.begin(), weights.end()};

  std::vector<Particle> newParticles;

  for (int i=0; i < num_particles; ++i) {
    int number = distribution(gen);
    newParticles.push_back(particles[number]);
  }

  particles.swap(newParticles);
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}