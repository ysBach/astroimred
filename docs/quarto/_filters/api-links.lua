-- Link documentation prose against the current quartodoc inventory.
local roles = {class = true, func = true, meth = true, obj = true, exc = true}

local function reference(text)
  local label, target = text:match("^(.-)%s*<([^>]+)>$")
  target = target or text:gsub("^~", "")
  return label or (text:sub(1, 1) == "~" and target:match("[^.]+$")) or text, target
end

local function index(items, project)
  local names, short = {}, {}
  local function add(table, key, uri)
    local previous = table[key]
    if previous == nil then
      table[key] = uri
    elseif previous ~= uri then
      table[key] = false -- Ambiguous, even if a third entry matches one of these.
    end
  end
  for _, item in ipairs(items) do
    if item.domain == "py" and (item.role == "class" or item.role == "exception"
        or item.role == "function" or item.role == "method") then
      local uri = item.uri:gsub("%$$", item.name)
      add(names, item.name, uri)
      local relative = item.name:sub(#project + 2)
      if item.name:sub(1, #project + 1) == project .. "." then
        if relative:find(".", 1, true) then
          add(names, relative, uri)
          local member = relative:match("^([A-Z][%w_]*%.[%w_]+)$")
            or relative:match("%.([A-Z][%w_]*%.[%w_]+)$")
          if member then add(names, member, uri) end
        end
        add(short, relative:match("[^.]+$"), uri)
      end
    end
  end
  return names, short
end

local function Pandoc(doc)
  local settings = doc.meta["api-links"] or {}
  local offset = quarto.project.offset
  local file = assert(io.open(offset .. "/objects.json", "r"),
    "Missing quartodoc inventory: generate the API reference before rendering")
  local inventory = quarto.json.decode(file:read("*a"))
  file:close()
  local names, short = index(inventory.items, inventory.project)

  local function resolve(text, explicit)
    local name = text:match("^([%a_][%w_.]*)%b()$") or text:match("^([%a_][%w_.]*)$")
    if not name then return nil end
    local prefix, rest = name:match("^([^.]+)%.(.+)$")
    if prefix then
      local alias = settings.aliases and settings.aliases[prefix]
      if alias then name = pandoc.utils.stringify(alias) .. "." .. rest end
      local reexport = settings.reexports and settings.reexports[name]
      if reexport then name = pandoc.utils.stringify(reexport) end
      return names[name]
    end
    -- Short common words can describe an algorithm, option, or parameter.
    -- Require a call, a conventional class/underscore name, or an explicit reference.
    if explicit or text ~= name or name:match("^[A-Z]") or name:find("_", 1, true) then
      return short[name]
    end
  end

  return doc:walk({
    traverse = "topdown",
    Link = function(link)
      -- Griffe's NumPy See Also section can emit an empty Markdown link with
      -- a Sphinx role as the target. Resolve it as a reference, not as a URL.
      local role, text = link.target:gsub("%%60", "`"):match("^:([%a:]+):`(.-)`$")
      if role and roles[role:gsub("^py:", "")] then
        local label, target = reference(text)
        local uri = resolve(target, true)
        if not uri then
          return #link.content > 0 and link.content or pandoc.Code(label), false
        end
        if #link.content == 0 then link.content = {pandoc.Code(label)} end
        link.target = offset .. "/" .. uri
      end
      return link, false
    end,
    Image = function(img) return img, false end,
    Header = function(header) return header, false end,
    Inlines = function(inlines)
      local out = pandoc.List()
      for _, el in ipairs(inlines) do
        if el.t == "Code" and not el.classes:includes("no-api-link") then
          local previous = out[#out]
          local role
          if previous and previous.t == "Str" then
            role = previous.text:match(":py:(%a+):$") or previous.text:match(":(%a+):$")
          end
          local target = el.attributes["api-target"]
          local label = el.text
          if roles[role] then
            label, target = reference(label)
          end
          local uri = resolve(target or el.text, target ~= nil)
          if el.attributes["api-target"] and not uri then
            error("Unknown or ambiguous API target: " .. target)
          end
          if uri then
            if roles[role] then
              previous.text = previous.text:gsub(":py:" .. role .. ":$", ""):gsub(":" .. role .. ":$", "")
              if previous.text == "" then out:remove(#out) end
            end
            el.attributes["api-target"] = nil
            el.text = label
            el = pandoc.Link(el, offset .. "/" .. uri)
          end
        end
        out:insert(el)
      end
      return out
    end,
  })
end

return {{ Pandoc = Pandoc }}
