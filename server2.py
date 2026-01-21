import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from xmlrpc import server
class JSONRPCHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length=int(self.headers['Content-Length'])
        body=self.rfile.read(content_length)
        request=json.loads(body)
        response={
            "jsonrpc":"2.0",
            "id":request.get("id")
        }

        try:
            method=request["method"]
            params=request.get("params", [])

            if method=="subtract":
                result=params[0]-params[1]
                response["result"]=result

            else:
                raise Exception("Method not found")
            
        except Exception as e:
            response["error"]={
                "code":-32601,
                "message":str(e)
            }

        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

def run():
    server=HTTPServer(("localhost",5000),JSONRPCHandler)
    print("Starting server on http://localhost:5000")
    server.serve_forever()
    
if __name__=="__main__":
    run()