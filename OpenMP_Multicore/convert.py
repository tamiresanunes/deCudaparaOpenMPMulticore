"""Regera o port C++ a partir dos fontes CUDA; exige apenas Python 3."""
from pathlib import Path
import re

source = Path(__file__).resolve().parent.parent / "c_cuda"
target = Path(__file__).resolve().parent


def matching(text, start, opening="(", closing=")"):
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == opening:
            depth += 1
        elif text[pos] == closing:
            depth -= 1
            if depth == 0:
                return pos
    raise ValueError(f"Unbalanced {opening} at {start}")


def convert_launches(code):
    pattern = re.compile(r"\b(\w+)<<<([^>]+)>>>\s*\(")
    while match := pattern.search(code):
        end = matching(code, match.end() - 1)
        config = [part.strip() for part in match.group(2).split(",")]
        args = code[match.end():end].strip()
        replacement = f"{match.group(1)}({config[0]}, {config[1]}"
        if args:
            replacement += ", " + args
        replacement += ")"
        code = code[:match.start()] + replacement + code[end + 1:]
    return code


def convert_kernels(code):
    pattern = re.compile(r"\b(?:static\s+)?__global__\s+void\s+(\w+)\s*\(")
    while match := pattern.search(code):
        params_end = matching(code, match.end() - 1)
        body_start = code.index("{", params_end)
        body_end = matching(code, body_start, "{", "}")
        original = code[match.start():body_end + 1]
        params = code[match.end():params_end].strip()
        body = code[body_start + 1:body_end]
        body = re.sub(r"blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x", "tid_x", body)
        body = re.sub(r"blockIdx\.y\s*\*\s*blockDim\.y\s*\+\s*threadIdx\.y", "tid_y", body)
        body = body.replace("threadIdx.x == 0 && blockIdx.x == 0", "tid_x == 0")
        prefix = "static " if original.lstrip().startswith("static") else ""
        if match.group(1) in ("calc_1", "calc_2", "calc_3"):
            stage = {"calc_1": "primeiro", "calc_2": "segundo", "calc_3": "terceiro"}[match.group(1)]
            description = f"Calcula o {stage} estágio da integração temporal em paralelo."
        else:
            description = "Executa a varredura " + match.group(1).replace("_", " ") + " em paralelo, preservando os índices da malha original."
        updated = (f"// {description}\n{prefix}void {match.group(1)}(dim3 grid, dim3 block"
                   + (", " + params if params else "") + ") {\n"
                   + "    cpu_for_each(grid, block, [&](int tid_x, int tid_y) {" + body + "\n    });\n}")
        code = code[:match.start()] + updated + code[body_end + 1:]
    return code


def convert(code, name):
    code = re.sub(r"#include\s*<(?:cuda_runtime\.h|cuda\.h|device_launch_parameters\.h|cub/cub\.cuh)>\s*", "", code)
    code = convert_kernels(code)
    code = convert_launches(code)
    code = code.replace("cudaMallocManaged", "cpuMalloc").replace("cudaMalloc", "cpuMalloc")
    code = code.replace("cudaFree", "cpuFree").replace("cudaMemcpy(", "cpuMemcpy(")
    code = re.sub(r",\s*cudaMemcpy(?:DeviceToHost|HostToDevice|DeviceToDevice)\s*\)", ")", code)
    code = code.replace("cudaDeviceSynchronize();", "")
    code = re.sub(r"cudaStream_t\s+([^;]+);", "", code)
    code = re.sub(r"cudaStream(?:Create|Synchronize|Destroy)\([^;]*\);", "", code)
    code = code.replace("cudaEvent_t", "CpuEvent")
    for suffix in ("Create", "Record", "Synchronize", "ElapsedTime", "Destroy"):
        code = code.replace("cudaEvent" + suffix, "cpuEvent" + suffix)
    code = code.replace("[CUDA]", "[OpenMP]")
    code = code.replace("kernel_solve_", "solve_")
    if name == "comum":
        code = re.sub(r"\s*cudaError_t status = cudaGetLastError\(\);.*?\n    }", "", code, flags=re.S)
    if name == "solve_P":
        code = re.sub(r"\s*cudaError_t calc3_error = cudaGetLastError\(\);.*?\n    }", "", code, flags=re.S)
        code = re.sub(r"\s*calc3_error =\s*\n\s*if\(calc3_error != cudaSuccess\).*?\n    }", "", code, flags=re.S)
    if name == "main":
        code = re.sub(r"\s*// NOTA: Este código usa Unified Memory.*?time_h2d = 0\.0f;", "\n        // Não há transferência host/dispositivo na versão multicore.\n        time_h2d = 0.0f;", code, flags=re.S)
        code = code.replace("CUDA Events", "Temporizadores OpenMP")
        code = code.replace("float time_h2d, time_d2h, time_timestep;", "float time_timestep;")
        code = "\n".join(line for line in code.split("\n") if not any(
            marker in line for marker in ("ev_h2d", "ev_d2h", "acc_h2d", "acc_d2h",
                                      "time_h2d", "time_d2h", "H2D", "D2H")))
    if name == "initial":
        # A extensão de domínio copia a última célula conhecida da mesma linha.
        # Ler j-1 durante a varredura paralela introduziria uma dependência entre threads.
        code = code.replace("dev_um[i*(jmax+1)+(j-1)]", "dev_umr[i*(rjmax+1)+rjmax]")
        code = code.replace("dev_vm[i*(jmax+2)+(j-1)]", "dev_vmr[i*(rjmax+2)+rjmax]")
        for field, previous in (("dev_p", "dev_pres"), ("dev_t", "dev_tr"),
                                ("dev_z", "dev_zr"), ("dev_h", "dev_hr")):
            code = code.replace(f"{field}[i*(jmax+1)+(j-1)]", f"{previous}[i*(rjmax+1)+rjmax]")
        code = code.replace('fopen("input/iterations.dat", "r+")', 'fopen("input/iterations.dat", "r")')
        code = code.replace('fopen("input/reference.dat", "r+")', 'fopen("input/reference.dat", "r")')
    if name == "bcC":
        # O original omitia parênteses e acessava índice negativo no canto direito.
        code = code.replace("imax-1*(jmax+1)", "(imax-1)*(jmax+1)")
    return code


DESCRIPTIONS = {
    "calcular": "Calcula dimensões da malha e parâmetros físicos a partir dos argumentos.",
    "alocar_globais": "Aloca as malhas e campos auxiliares compartilhados pelo solver.",
    "desalocar_globais": "Libera todos os campos globais ao encerrar a simulação.",
    "grid_2d": "Calcula a geometria da varredura bidimensional com blocos de 16 pontos.",
    "grid_1d": "Calcula quantos blocos cobrem uma dimensão da malha.",
    "main": "Lê os argumentos, inicializa a malha e executa passos físicos e iterações de pseudotempo.",
    "init": "Lê parâmetros de iteração e propriedades de referência dos arquivos de entrada.",
    "IC": "Define os campos iniciais de velocidade, pressão, temperatura e concentração.",
    "restart": "Carrega uma solução anterior para continuar na mesma malha.",
    "restart_dom": "Carrega uma solução anterior e estende seus campos para um domínio maior.",
    "mesh": "Constrói a malha e classifica as células de fluido, obstáculo e contorno.",
    "output": "Escreve os campos calculados e arquivos de reinício.",
    "transient": "Exporta campos de um passo físico para visualização temporal.",
    "convergence": "Registra os resíduos para acompanhar a convergência.",
    "probe": "Encontra a posição de mudança de sinal da velocidade e a registra.",
    "flametip": "Estima e registra a posição da ponta de chama.",
    "max_reduce": "Retorna o maior valor absoluto dos resíduos por redução OpenMP.",
    "calcula_x_y": "Gera as coordenadas principais do domínio.",
    "calcula_xm_ym": "Gera as coordenadas deslocadas usadas nas faces dos volumes.",
    "calcula_vol": "Calcula volumes das células de velocidade e pressão.",
    "calcula_area_das_fases": "Calcula as áreas das faces dos volumes de controle.",
    "calcula_dx_dy": "Calcula espaçamentos locais da malha nas duas direções.",
    "comp_mean": "Interpola velocidades das faces para os centros das células.",
    "upwind_U_pair": "Calcula o transporte nas bordas do campo horizontal de velocidade.",
    "upwind_V_pair": "Calcula o transporte nas bordas do campo vertical de velocidade.",
    "RESU": "Monta o resíduo da equação de momento horizontal.",
    "RESV": "Monta o resíduo da equação de momento vertical.",
    "RESZ": "Monta o resíduo da equação de energia.",
    "RESC": "Monta o resíduo da equação de concentração.",
    "solve_U": "Integra a velocidade horizontal em três estágios e retorna seu maior resíduo.",
    "solve_V": "Integra a velocidade vertical em três estágios e retorna seu maior resíduo.",
    "solve_P": "Atualiza a pressão por compressibilidade artificial e retorna seu maior resíduo.",
    "solve_Z": "Integra a equação de energia em três estágios.",
    "solve_C": "Integra a equação de concentração em três estágios.",
    "bcUV": "Aplica condições de contorno das duas componentes de velocidade.",
    "bcP": "Aplica condições de contorno da pressão, deixando os cantos para o final.",
    "bcZ": "Aplica condições de contorno da energia, deixando os cantos para o final.",
    "bcC": "Aplica condições de contorno da concentração, deixando os cantos para o final.",
}


def document_functions(code, filename):
    pattern = re.compile(r"^(?:static )?(?:void|double|int|dim3) (\w+)\s*\(", re.M)
    def add_comment(match):
        name = match.group(1)
        if name.startswith(("calc_", "contorno_", "atualiza_", "atualizar_", "restart_dom_", "pontos_medios")):
            return match.group(0)  # Comentários dos kernels já foram inseridos acima.
        description = DESCRIPTIONS.get(name, "Executa a etapa " + name.replace("_", " ") + " do solver.")
        return "// " + description + "\n" + match.group(0)
    return pattern.sub(add_comment, code)


for path in source.glob("*.cu"):
    if path.stem == "max_reduce":
        continue
    text = document_functions(convert(path.read_text(encoding="utf-8"), path.stem), path.name)
    (target / (path.stem + ".cpp")).write_text(text, encoding="utf-8")

for name in ("comum.h", "functions.h"):
    text = (source / name).read_text(encoding="utf-8")
    text = re.sub(r"#include\s*<(?:cuda_runtime\.h|cuda\.h|device_launch_parameters\.h)>\s*", "", text)
    if name == "comum.h":
        text = text.replace('#include "functions.h"', '#include "functions.h"\n#include "cpu_runtime.h"')
        text = text.replace("struct Iterations{", "// Parâmetros de iteração e critérios de convergência lidos da entrada.\nstruct Iterations{")
        text = text.replace("struct Ref{", "// Propriedades de referência utilizadas na adimensionalização.\nstruct Ref{")
    else:
        text = text.replace('#include "comum.h"\n', '')
    (target / name).write_text(text, encoding="utf-8")

for path in source.glob("*.c"):
    if path.name in ("probe.c", "convergence.c", "flametip.c", "transient.c"):
        text = document_functions(path.read_text(encoding="utf-8"), path.name)
        (target / path.name).write_text(text, encoding="utf-8")
