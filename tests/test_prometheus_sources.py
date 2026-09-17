import unittest
from sma import Config


class PrometheusSourcesTest(unittest.TestCase):
    def config(self, services, measurements):
        return Config.from_dict({'sma': {'services': services, 'measurements': measurements}})

    def test_legacy_default(self):
        config = self.config({'prometheus': {'address': 'http://cluster'}}, [{'cpu': {'query': 'cpu'}}])
        query = config.measurement_queries()['cpu']
        self.assertIs(query.client, config.prometheus_client())
        self.assertEqual(query.client.base_url, 'http://cluster')

    def test_named_source_and_default_use_different_clients(self):
        config = self.config({'prometheus': {'address': 'http://cluster'},
                              'zodiac': {'type': 'prometheus', 'address': 'http://zodiac'}},
                             [{'cpu': {'query': 'cpu'}}, {'jobs': {'query': 'jobs', 'service': 'zodiac'}}])
        queries = config.measurement_queries()
        self.assertEqual(queries['cpu'].client.base_url, 'http://cluster')
        self.assertEqual(queries['jobs'].client.base_url, 'http://zodiac')
        self.assertIs(config.create_measurement_query(config.measurements['jobs']).client, queries['jobs'].client)

    def test_named_source_without_default(self):
        config = self.config({'zodiac': {'type': 'prometheus', 'address': 'http://zodiac'}},
                             [{'jobs': {'query': 'jobs', 'service': 'zodiac'}}])
        self.assertEqual(config.measurement_queries()['jobs'].client.base_url, 'http://zodiac')

    def test_unknown_source_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown service'):
            self.config({'prometheus': {'address': 'http://cluster'}},
                        [{'jobs': {'query': 'jobs', 'service': 'typo'}}])

    def test_unknown_type_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown service type'):
            self.config({'zodiac': {'type': 'typo', 'address': 'http://zodiac'}}, [])


if __name__ == '__main__':
    unittest.main()
