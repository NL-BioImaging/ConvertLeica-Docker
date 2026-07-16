import os
import json
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import webbrowser
import threading
from ci_leica_converters_helpers import read_leica_file,get_image_metadata,get_image_metadata_LOF
from CreatePreview import create_preview_base64_image, create_preview_image
from leica_converter import convert_leica, require_conversion_dependencies
import sys
import tempfile


ROOT_DIR = "L:/Archief/active/cellular_imaging/OMERO_test"  # change as needed
OUTPUT_SUBFOLDER = "_c"  # Output subfolder for converted files
DEFAULT_PORT = 8000  # Default port for the server
MAX_XY_SIZE = 3192 # Maximum XY size of OME_Tiff files without pyramids
PREVIEW_SIZE = 200 # Default preview size in pixels
PREVIEW_STEPS = [24, 100, 200]  # Progressive preview steps
PREVIEW_CACHE_MAX = 500  # Maximum number of cached previews

def get_cache_dir():
    d = os.path.join(tempfile.gettempdir(), "leica_preview_cache")
    os.makedirs(d, exist_ok=True)
    return d

class SSEStream:
    """
    Server-Sent Events (SSE) stream helper for sending progress updates to the client.
    """
    def __init__(self, wfile):
        self.wfile = wfile
        self.line_buffer = ""
    def write(self, chunk):
        if not self.wfile:
            return
        self.line_buffer += chunk
        while '\n' in self.line_buffer:
            line, self.line_buffer = self.line_buffer.split('\n', 1)
            if line.strip():
                msg = json.dumps({"type":"progress","message":line})
                sse = f"data: {msg}\n\n"
                try:
                    self.wfile.write(sse.encode())
                    self.wfile.flush()
                except:
                    self.wfile = None
    def flush(self):
        if self.line_buffer.strip() and self.wfile:
            msg = json.dumps({"type":"progress","message":self.line_buffer.strip()})
            sse = f"data: {msg}\n\n"
            try:
                self.wfile.write(sse.encode())
                self.wfile.flush()
            except:
                self.wfile = None
        self.line_buffer = ""

class MyHTTPRequestHandler(SimpleHTTPRequestHandler):
    """
    Custom HTTP request handler for the Leica conversion web server.
    Handles API endpoints for listing files, previewing images, configuration, and conversion.
    """

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/"):
            if parsed.path == "/api/list":
                self.handle_list(parsed.query)
            elif parsed.path == "/api/config":
                self.handle_config()
            else:
                self.send_response(404)
                self.end_headers()
            return
        # Else serve files as usual.
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/preview":
            self.handle_preview()
        elif parsed.path == "/api/lof_metadata":
            self.handle_lof_metadata()
        elif parsed.path == "/api/convert_leica":
            self.handle_convert_leica()
        elif parsed.path == "/api/preview_status":
            self.handle_preview_status()
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def handle_list(self, query):
        params = urllib.parse.parse_qs(query)
        # If no "dir" provided, use the ROOT_DIR.
        directory = params.get("dir", [ROOT_DIR])[0]
        directory = os.path.normpath(directory)
        folder_uuid = params.get("folder_uuid", [None])[0]  # Get folder_uuid from query
        
        response = {"items": []}
        try:
            ext = os.path.splitext(directory)[1].lower()
            if not os.path.isdir(directory) and ext in (".lif", ".xlef"):
                if folder_uuid:
                    folder_metadata = read_leica_file(directory, folder_uuid=folder_uuid)  # Pass folder_uuid
                else:
                    folder_metadata = read_leica_file(directory)
                try:
                    parsed_dict = json.loads(folder_metadata)
                    if "children" in parsed_dict:
                        for child in parsed_dict["children"]:
                            name = child.get("name", "").lower()
                            if ("_environmentalgraph" in name or 
                                name.endswith(".lifext") or 
                                name.lower in ["iomanagerconfiguation", "iomanagerconfiguration", "IOManagerConfiguation"]):
                                continue
                            if "path" not in child:
                                child["path"] = directory
                            response["items"].append(child)
                    else:
                        response["items"] = [parsed_dict]
                    response["folder_metadata"] = folder_metadata  # Pass folder_metadata to client
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e}")
                    response = folder_metadata
            else:
                # List directory items.
                all_items = os.listdir(directory)
                # If at least one .xlef file exists, filter to only .xlef files.
                if any(os.path.splitext(n)[1].lower() == ".xlef" for n in all_items):
                    items_to_list = [n for n in all_items if os.path.splitext(n)[1].lower() == ".xlef"]
                else:
                    items_to_list = all_items

                for name in items_to_list:
                    lowname = name.lower()
                    if ("metadata" in lowname or "_pmd_" in lowname or "_histo" in lowname or
                        "_environmetalgraph" in lowname or lowname.endswith(".lifext") or
                        lowname in ["iomanagerconfiguation", "iomanagerconfiguration"]):
                        continue
                    abs_path = os.path.join(directory, name)
                    if os.path.isdir(abs_path) or os.path.splitext(name)[1].lower() in (".lif", ".xlef"):
                        item_type = "Folder"
                    else:
                        item_type = "File"
                    response["items"].append({
                        "name": name,
                        "path": abs_path,
                        "type": item_type
                    })
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        except Exception as e:
            self.send_error(500, str(e))

    def handle_lof_metadata(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode('utf-8'))
            filePath = data.get("filePath")

            if not filePath:
                self.send_error(400, "Missing filePath parameter")
                return

            try:
                metadata = read_leica_file(filePath)
                metadata = json.loads(metadata) # Parse the metadata string into a JSON object
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(metadata).encode("utf-8")) # Send the JSON object
            except Exception as e:
                self.send_error(500, str(e))

        except Exception as e:
            self.send_error(500, str(e))

    def handle_preview(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode('utf-8'))
            filePath = data.get("filePath")
            image_uuid = data.get("image_uuid")
            folder_metadata = data.get("folder_metadata")
            preview_height = data.get("preview_height", 256)  # Default to 256 if not provided

            ext = os.path.splitext(filePath)[1].lower()
            if ext == ".lof":
                image_metadata = read_leica_file(filePath)
            elif ext == ".xlef": # If xlef file, get image metadata("save_child_name") from folder_metadata
                image_metadata_f = json.loads(get_image_metadata(folder_metadata, image_uuid))
                image_metadata = json.loads(get_image_metadata_LOF(folder_metadata, image_uuid))
                if "save_child_name" in image_metadata_f:
                    image_metadata["save_child_name"] = image_metadata_f["save_child_name"]
                image_metadata = json.dumps(image_metadata)  # Convert back to JSON string
            else:
                # For .lif, fetch full metadata (includes Position, bytes offsets, etc.)
                image_metadata = read_leica_file(filePath, image_uuid=image_uuid)
            # Use server-side cache to create or reuse preview file, then return base64
            cache_dir = get_cache_dir()
            # Detect whether this preview already exists in cache (for client hint)
            try:
                meta_for_uuid = json.loads(image_metadata) if isinstance(image_metadata, str) else image_metadata
                uid = meta_for_uuid.get("UniqueID")
            except Exception:
                uid = None
            cache_path = None
            cached_before = False
            if uid:
                cache_filename = f"{uid}_h{int(preview_height)}.png"
                cache_path = os.path.join(cache_dir, cache_filename)
                cached_before = os.path.exists(cache_path)

            # Create (or reuse) cached preview
            cached_file = create_preview_image(image_metadata, cache_dir, preview_height=int(preview_height), use_memmap=True, max_cache_size=PREVIEW_CACHE_MAX)
            # Read file and return base64 data URL
            with open(cached_file, 'rb') as f:
                b64 = f.read()
            mime = "image/png"
            import base64 as _b64
            src = f"data:{mime};base64,{_b64.b64encode(b64).decode('utf-8')}"

            # Return both preview src and image metadata
            response = {"src": src, "metadata": image_metadata, "height": int(preview_height), "cached": bool(cached_before)}
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        except Exception as e:
            self.send_error(500, str(e))


    def handle_convert_leica(self):
        sse = None                               # initialize
        orig_stdout = sys.stdout
        content_length = int(self.headers['Content-Length'])
        post = self.rfile.read(content_length)

        try:
            require_conversion_dependencies("ome-tiff")
            data = json.loads(post.decode())
            inp = data.get("filePath")
            uuid_ = data.get("image_uuid")
            if not inp or not uuid_:
                self.send_response(400)
                self.send_header("Content-Type","application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success":False,"error":"Missing parameters"}).encode())
                return

            self.send_response(200)
            self.send_header("Content-Type","text/event-stream")
            self.send_header("Cache-Control","no-cache")
            self.send_header("Connection","keep-alive")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()

            sse = SSEStream(self.wfile)
            sys.stdout = sse

            # determine output folder
            outdir = os.path.join(os.path.dirname(inp), OUTPUT_SUBFOLDER)
            os.makedirs(outdir, exist_ok=True)

            # call converter
            result_json = convert_leica(
                inputfile=inp,
                image_uuid=uuid_,
                outputfolder=outdir,
                show_progress=True
            )
            sse.flush()

            # Parse and log the JSON result to the SSE progress stream (like GUI log)
            try:
                result = json.loads(result_json)
            except Exception:
                result = []
            try:
                pretty = json.dumps(result if result else result_json, indent=2, ensure_ascii=False)
                print("Conversion result (JSON):\n" + pretty)
            except Exception:
                # Best effort: print the raw string
                try:
                    print("Conversion result (JSON):\n" + str(result_json))
                except Exception:
                    pass
            sse.flush()
            payload = {"type":"result","payload":{"success":bool(result),"result":result}}
            if sse.wfile:
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

        except Exception as e:
            if sse:
                sse.flush()
            err = {"type":"error","message":str(e)}
            if sse and sse.wfile:
                self.wfile.write(f"data: {json.dumps(err)}\n\n".encode())
                self.wfile.flush()
        finally:
            sys.stdout = orig_stdout
            if sse and sse.wfile:
                self.wfile.write(f"data: {json.dumps({'type':'end'})}\n\n".encode())
                self.wfile.flush()

    def handle_config(self):
        # return ROOT_DIR and constants to client
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({
            "rootDir": ROOT_DIR,
            "maxXYSize": MAX_XY_SIZE,
            "previewSize": PREVIEW_SIZE,
            "previewSteps": PREVIEW_STEPS,
            "previewCacheMax": PREVIEW_CACHE_MAX
        }).encode("utf-8"))

    def handle_preview_status(self):
        # Returns max cached preview height and image dimensions, without generating previews
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode('utf-8'))
            filePath = data.get("filePath")
            image_uuid = data.get("image_uuid")
            folder_metadata = data.get("folder_metadata")

            ext = os.path.splitext(filePath)[1].lower()
            if ext == ".lof":
                image_metadata = read_leica_file(filePath)
                meta = json.loads(image_metadata)
            elif ext == ".xlef":
                image_metadata_f = json.loads(get_image_metadata(folder_metadata, image_uuid))
                lof_like = json.loads(get_image_metadata_LOF(folder_metadata, image_uuid))
                if "save_child_name" in image_metadata_f:
                    lof_like["save_child_name"] = image_metadata_f["save_child_name"]
                meta = lof_like
            else:
                meta = json.loads(get_image_metadata(folder_metadata, image_uuid))

            uid = meta.get("UniqueID")
            xs = meta.get("xs") or (meta.get("dimensions") or {}).get("x")
            ys = meta.get("ys") or (meta.get("dimensions") or {}).get("y")

            max_cached = 0
            if uid:
                cache_dir = get_cache_dir()
                for h in PREVIEW_STEPS:
                    p = os.path.join(cache_dir, f"{uid}_h{int(h)}.png")
                    if os.path.exists(p):
                        max_cached = max(max_cached, int(h))

            resp = {"maxCached": max_cached, "xs": xs, "ys": ys}
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        except Exception as e:
            self.send_error(500, str(e))

def run(server_class=ThreadingHTTPServer, handler_class=MyHTTPRequestHandler, port=DEFAULT_PORT):
    require_conversion_dependencies("ome-tiff")
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on http://localhost:{port}")
    # Launch default browser without blocking server startup
    try:
        url = f"http://localhost:{port}"
        if os.name == 'nt':
            try:
                os.startfile(url)  # type: ignore[attr-defined]
            except Exception:
                threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
        else:
            threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    except Exception:
        pass
    httpd.serve_forever()

if __name__ == "__main__":
    run()
