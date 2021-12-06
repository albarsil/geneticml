import unittest
import flask
import json

from flask.globals import request

from pyschemavalidator import validate_param


class TestExpects(unittest.TestCase):
    def setUp(self):
        self.app = flask.Flask(__name__)

        @self.app.route('/')
        def no_schema():
            return 'happy'

        @self.app.route('/case1')
        @validate_param(key="name", keytype=str, isrequired=True)
        @validate_param(key="price", keytype=float, isrequired=False)
        def case1():
            return 'worked'

        @self.app.route('/case2')
        @validate_param(key="name", keytype=str, isrequired=True)
        @validate_param(key="price", keytype=float, isrequired=True)
        def case2():
            return 'worked'

        @self.app.route('/case3')
        @validate_param(key="elements", keytype=list, isrequired=True, innertype=str, constrain=["a", "b"])
        def case3():
            return 'worked'

        @self.app.route('/case4')
        @validate_param(key="price", keytype=float, isrequired=True, maxmin=(0,4))
        def case4():
            return 'worked'

        @self.app.route('/case5')
        @validate_param(key="tag", keytype=int, isrequired=True, constrain=[1,2,3])
        def case5():
            return 'worked'

        self.client = self.app.test_client()
        self.ctx = self.app.app_context()
        self.ctx.push()

    def tearDown(self):
        self.ctx.pop()

    def test_case1(self):
        response = self.client.get(
            '/case1',
            data=json.dumps({"name": "Eggs", "price": 34.99}),
            content_type='application/json'
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual('Eggs', flask.g.data['name'])
        self.assertEqual(34.99, flask.g.data['price'])

    def test_case1_wrong(self):
        response = self.client.get(
            '/case1',
            data=json.dumps({"name": "Eggs", "price": "34.99"}),
            content_type='application/json'
        )
        self.assertEqual(400, response.status_code)

    def test_case2(self):
        response = self.client.get(
            '/case1',
            data=json.dumps({"name": "Eggs"}),
            content_type='application/json'
        )
        self.assertEqual(400, response.status_code)

    def test_case3(self):
        response = self.client.get(
            '/case3',
            data=json.dumps({"elements": ["a", "b"]}),
            content_type='application/json'
        )
        self.assertEqual(200, response.status_code)

    def test_case3_wrong(self):
        response = self.client.get(
            '/case3',
            data=json.dumps({"elements": ["a", "c"]}),
            content_type='application/json'
        )
        self.assertEqual(400, response.status_code)

    def test_case4(self):
        response = self.client.get(
            '/case4',
            data=json.dumps({"price": 3.12}),
            content_type='application/json'
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual(3.12, flask.g.data['price'])

    def test_case4_wrong(self):
        response = self.client.get(
            '/case4',
            data=json.dumps({"price": 15}),
            content_type='application/json'
        )
        self.assertEqual(400, response.status_code)

    def test_case5(self):
        response = self.client.get(
            '/case5',
            data=json.dumps({"tag": 3}),
            content_type='application/json'
        )
        self.assertEqual(200, response.status_code)

    def test_case5_wrong(self):
        response = self.client.get(
            '/case5',
            data=json.dumps({"tag": 2.15}),
            content_type='application/json'
        )
        self.assertEqual(400, response.status_code)

    # def test_validation_invalid(self):
    #     response = self.client.get('/schema',
    #                                data='{"name": "Eggs", "price": "invalid"}',
    #                                content_type='application/json')
    #     self.assertEqual(400, response.status_code)
    #     self.assertIn('is not of type \'number\'', response.data.decode())

    # def test_missing_parameter(self):
    #     response = self.client.get('/schema',
    #                                data='{"name": "Eggs"}',
    #                                content_type='application/json')
    #     self.assertEqual(400, response.status_code)
    #     self.assertIn('is a required property', response.data.decode())

    # def test_additional_parameter(self):
    #     response = self.client.get('/schema',
    #                                data='{"name": "Eggs", "price": 34.99}',
    #                                content_type='application/json')
    #     self.assertEqual(200, response.status_code)
