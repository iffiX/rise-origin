# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Static registry of Python authored MCP tools.

A tool is declared with ``@mcp.tool`` anywhere, before any server exists; a server
component serves every declared tool by calling :meth:`MCPToolRegistry.register_all`
when it starts. A ``dict`` or ``str`` returned by a tool is sent as text, a
``{"_mcp_image": {"data_base64": ..., "mime": ...}}`` dict as an image, and a raised
exception is caught and returned as an error.

Example::

    from rise_origin.runtime.mcp import mcp

    @mcp.tool(
        "spawn_box",
        "Add a box at (x, y).",
        schema={
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"],
        },
    )
    def spawn_box(handle, x, y):
        handle.add_entity(...)
        return {"ok": True}
"""

import functools
import json

__all__ = ["mcp", "MCPToolRegistry"]


class MCPToolRegistry:
    """Collects declared tools and registers them onto a server component."""

    def __init__(self) -> None:
        self._tools = []

    def tool(self, name: str, description: str, schema: dict):
        """Declare the decorated function as an MCP tool.

        The function is called with the full world handle and the tool arguments as
        keyword arguments, and its return value becomes the MCP result.

        Args:
            name: Tool name, unique across the tool set.
            description: One line shown to the client.
            schema: JSON Schema of the arguments, as a dict.

        Returns:
            The decorator that records and returns the function unchanged.
        """

        def declare(function):
            self._tools.append((name, description, schema, function))
            return function

        return declare

    def register_all(self, component) -> None:
        """Register every declared tool onto a started server component.

        Args:
            component: The ``RS_WorldSessionComponentMCP`` to serve the tools.
        """
        for name, description, schema, function in self._tools:
            component.register_tool(
                name,
                description,
                json.dumps(schema),
                functools.partial(MCPToolRegistry._run, function),
            )

    @staticmethod
    def _run(function, handle, arguments: str) -> str:
        """Invoke a tool function and format its outcome as an MCP result json."""
        try:
            return MCPToolRegistry._format(function(handle, **json.loads(arguments)))
        except Exception as error:
            return MCPToolRegistry._error(str(error))

    @staticmethod
    def _format(result) -> str:
        """Turn a tool return value into an MCP result json string."""
        if isinstance(result, dict) and "_mcp_image" in result:
            image = result["_mcp_image"]
            content = [
                {"type": "image", "data": image["data_base64"], "mimeType": image["mime"]}
            ]
        else:
            text = result if isinstance(result, str) else json.dumps(result)
            content = [{"type": "text", "text": text}]
        return json.dumps({"content": content})

    @staticmethod
    def _error(message: str) -> str:
        """Turn an error message into an MCP error result json string."""
        return json.dumps({"content": [{"type": "text", "text": message}], "isError": True})


mcp = MCPToolRegistry()
