import unittest

from tinyCode.mcp.protocol import (
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    decode_message,
)


class MCPProtocolTests(unittest.TestCase):
    def test_decode_message_rejects_malformed_json_rpc_fields(self):
        bad_frames = [
            '{"jsonrpc":"1.0","id":1,"method":"tools/list"}',
            '{"jsonrpc":"2.0","id":1,"method":123}',
            '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":[]}',
            '{"jsonrpc":"2.0","method":123}',
            '{"jsonrpc":"2.0","method":"notifications/ready","params":"bad"}',
            '{"jsonrpc":"2.0","id":1,"error":"bad"}',
        ]

        for frame in bad_frames:
            with self.subTest(frame=frame):
                self.assertIsNone(decode_message(frame))

    def test_decode_message_accepts_valid_json_rpc_messages(self):
        request = decode_message(
            '{"jsonrpc":"2.0","id":"req-1","method":"tools/list","params":{"cursor":"a"}}'
        )
        notification = decode_message(
            '{"jsonrpc":"2.0","method":"notifications/ready"}'
        )
        response = decode_message(
            '{"jsonrpc":"2.0","id":"req-1","result":{"tools":[]}}'
        )

        self.assertIsInstance(request, JSONRPCRequest)
        self.assertEqual("tools/list", request.method)
        self.assertEqual({"cursor": "a"}, request.params)
        self.assertEqual("req-1", request.id)
        self.assertIsInstance(notification, JSONRPCNotification)
        self.assertEqual("notifications/ready", notification.method)
        self.assertIsInstance(response, JSONRPCResponse)
        self.assertEqual({"tools": []}, response.result)


if __name__ == "__main__":
    unittest.main()
