module not_test (a, b, c, d, e);
input a;
output b;
input [3:0] c;
output [3:0] d;
output e;
wire a;
wire b;
wire [3:0] c;
wire [3:0] d;
wire e;

assign b = ~(a);
assign d = ~(c);
assign e = ~( \escaped_name );
endmodule
